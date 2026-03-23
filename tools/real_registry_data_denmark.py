"""
Real Danish registry data from Statistikbanken and DST APIs.

REFERENCE MODULE — not imported by the ResearchClaw pipeline.
Use this as a standalone script or reference for understanding the APIs.
Generated experiment code should use urllib.request (stdlib), not requests.

Data sources:
1. NAME FREQUENCY (birth cohort): DST NavneBarometer API
   https://www.dst.dk/da/DstDk-Global/Udvikler/HostNavneBarometer?ajax=1
   Coverage: 1985-2024 (newborn name frequency by birth year)
   NOTE: DST does not provide name frequency data before 1985.

2. TOTAL MALE BIRTHS per year: Statistikbanken table FOD
   https://api.statbank.dk/v1/data (table: FOD)
   Coverage: 1973-present

3. EDUCATIONAL ATTAINMENT by age group: Statistikbanken table HFUDD11
   Coverage: 2008-present (by 5-year age group)
   We map birth_year → age group at time of measurement

IMPORTANT LIMITATIONS:
- Name-frequency-by-birth-year for 1975-1984 is NOT available via any public DST API.
  The NavneBarometer explicitly starts at 1985. We raise an explicit error for these years.
- Education/income by specific FIRST NAME is NOT in any public Statistikbanken table.
  Such linkage requires Danmarks Statistik microdata access (not public).
  We provide population-level education rates by birth cohort (age group proxy).
- Income by first name is similarly unavailable — we provide population median by birth cohort.

Usage (standalone, outside sandbox):
    python tools/real_registry_data_denmark.py

Author: AutoResearchClaw subagent (2026-03-22)
Moved to tools/ (2026-03-23): was incorrectly placed in researchclaw/data/
which is a pipeline-internal package. This module uses requests+pandas
which are NOT available in the experiment sandbox.
"""

import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
import time

try:
    import requests
except ImportError as _e:
    raise ImportError(
        "This standalone reference script requires 'requests'. "
        "Install it with: pip install requests\n"
        "Generated experiment code uses urllib.request (stdlib) instead."
    ) from _e

logger = logging.getLogger(__name__)

# DST API endpoints
NAVNEBAROMETER_URL = "https://www.dst.dk/da/DstDk-Global/Udvikler/HostNavneBarometer?ajax=1"
STATBANK_DATA_URL = "https://api.statbank.dk/v1/data"

# Years covered by the NavneBarometer
NAVN_BARO_START_YEAR = 1985
NAVN_BARO_END_YEAR = 2024

# Target names for the study
DEFAULT_TARGET_NAMES = ["Simon", "Lars", "Morten", "Thomas", "Martin", "Mikkel"]


def fetch_name_frequency_births(names: List[str], session: Optional[requests.Session] = None) -> Dict[str, List[int]]:
    """
    Fetch raw birth counts per year (1985-2024) for given male first names.

    Uses the DST NavneBarometer API.

    Args:
        names: List of Danish first names (case-insensitive)
        session: Optional requests.Session

    Returns:
        Dict mapping name → list of 40 birth counts (1985..2024)

    Raises:
        RuntimeError: if API returns an error or unexpected format
    """
    if session is None:
        session = requests.Session()

    result = {}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.dst.dk/da/Statistik/emner/borgere/navne/navne-til-nyfoedte",
        "User-Agent": "AutoResearchClaw/1.0 (research; contact research@voynich.ai)",
    }

    for name in names:
        resp = session.post(
            NAVNEBAROMETER_URL,
            data={"name": name},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 204:
            raise RuntimeError(f"DST NavneBarometer: no data found for name '{name}'")
        resp.raise_for_status()

        data = resp.json()
        # Response format: {"SIMON (M)": [count1985, count1986, ..., count2024], "SIMON (K)": [...]}
        male_key = next((k for k in data if "(M)" in k), None)
        if male_key is None:
            raise RuntimeError(f"DST NavneBarometer: no male entry in response for '{name}'. Keys: {list(data.keys())}")

        counts = data[male_key]
        expected_len = NAVN_BARO_END_YEAR - NAVN_BARO_START_YEAR + 1
        if len(counts) != expected_len:
            raise RuntimeError(
                f"DST NavneBarometer: expected {expected_len} years for '{name}', got {len(counts)}"
            )

        result[name.upper()] = counts
        logger.info(f"Fetched NavneBarometer for {name}: {counts[:5]}... (total {sum(counts)} across all years)")
        time.sleep(0.3)  # be polite

    return result


def fetch_total_male_births(years: List[int], session: Optional[requests.Session] = None) -> Dict[int, int]:
    """
    Fetch total male live births per year from Statistikbanken table FOD.

    Coverage: 1973-present

    Args:
        years: List of integer years
        session: Optional requests.Session

    Returns:
        Dict mapping year → total male births
    """
    if session is None:
        session = requests.Session()

    str_years = [str(y) for y in years]
    payload = {
        "table": "FOD",
        "format": "CSV",
        "delimiter": "Semicolon",
        "variables": [
            {"code": "BARNKON", "values": ["D"]},  # D = Boys
            {"code": "Tid", "values": str_years},
        ],
    }
    resp = session.post(STATBANK_DATA_URL, json=payload, timeout=20)
    resp.raise_for_status()

    # Strip BOM if present, normalize line endings
    text = resp.text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.strip().split("\n")
    header = lines[0].split(";")
    # Columns: BARNKON; TID; INDHOLD
    tid_col = header.index("TID")
    val_col = header.index("INDHOLD")

    result = {}
    for line in lines[1:]:
        parts = line.split(";")
        year = int(parts[tid_col])
        count = int(parts[val_col])
        result[year] = count

    return result


def fetch_tertiary_education_by_age(
    age_group: str,
    obs_years: List[int],
    sex: str = "M",
    session: Optional[requests.Session] = None,
) -> Dict[int, float]:
    """
    Fetch tertiary education rate (%) for a specific age group from HFUDD11.

    Tertiary education = H40 (Short cycle) + H5097 (Medium, unspec) + H6097 (Bachelor, unspec)
                       + H7097 (Master, unspec) + H8097 (PhD, unspec)

    Args:
        age_group: e.g. "30-34", "35-39", "40-44", "45-49"
        obs_years: Observation years to query
        sex: "M" or "K"
        session: Optional requests.Session

    Returns:
        Dict mapping observation_year → tertiary_education_pct
    """
    if session is None:
        session = requests.Session()

    tertiary_codes = ["H40", "H5097", "H6097", "H7097", "H8097"]

    payload = {
        "table": "HFUDD11",
        "format": "CSV",
        "delimiter": "Semicolon",
        "variables": [
            {"code": "BOPOMR", "values": ["000"]},         # Whole country
            {"code": "HERKOMST", "values": ["TOT"]},        # All origins
            {"code": "HFUDD", "values": ["TOT"] + tertiary_codes},
            {"code": "ALDER", "values": [age_group]},
            {"code": "KØN", "values": [sex]},
            {"code": "Tid", "values": [str(y) for y in obs_years]},
        ],
    }
    resp = session.post(STATBANK_DATA_URL, json=payload, timeout=20)
    resp.raise_for_status()

    # Strip BOM if present, normalize line endings
    text = resp.text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.strip().split("\n")
    header = lines[0].split(";")
    hfudd_col = header.index("HFUDD")
    tid_col = header.index("TID")
    val_col = header.index("INDHOLD")

    totals = {}
    tertiary_sums = {}

    for line in lines[1:]:
        parts = line.split(";")
        edu = parts[hfudd_col].split(" ")[0]  # e.g. "H40" from "H40 Korte..."
        year = int(parts[tid_col])
        value = int(parts[val_col]) if parts[val_col].strip() else 0

        raw_hfudd = parts[hfudd_col]
        if raw_hfudd.strip() == "I alt" or edu == "TOT":
            # "I alt" = total (Danish for "In total")
            totals[year] = value
        elif any(edu.startswith(tc) for tc in tertiary_codes):
            tertiary_sums[year] = tertiary_sums.get(year, 0) + value

    result = {}
    for year in obs_years:
        total = totals.get(year, 0)
        tertiary = tertiary_sums.get(year, 0)
        if total > 0:
            result[year] = 100.0 * tertiary / total
        else:
            result[year] = float("nan")

    return result


def birth_year_to_age_group(birth_year: int, obs_year: int = 2020) -> str:
    """
    Map birth year to a 5-year age group as observed in a given year.

    Args:
        birth_year: Year of birth
        obs_year: Year of observation (default 2020)

    Returns:
        Age group string like "35-39"
    """
    age = obs_year - birth_year
    age_groups = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69"]
    for grp in age_groups:
        lo, hi = map(int, grp.split("-"))
        if lo <= age <= hi:
            return grp
    return "TOT"  # fallback


def fetch_registry_data(
    names: List[str] = DEFAULT_TARGET_NAMES,
    years: Optional[List[int]] = None,
    data_end_year: int = 1995,
    data_start_year: int = 1985,
) -> pd.DataFrame:
    """
    Build a DataFrame of real Danish registry data from public APIs.

    The DataFrame has one row per (birth_year, name) combination and includes:
    - birth_year: Year of birth cohort
    - name: First name
    - frequency_pct: Percentage of male births with this name in that year
    - birth_count: Raw count of male babies with this name
    - total_male_births: Total male births that year
    - education_tertiary_pct: Population-level tertiary education rate for birth cohort
    - cohort_id: Unique identifier

    NOTE: education/income BY NAME is not available via public API.
    The education_tertiary_pct column reflects the POPULATION-LEVEL rate for the 
    birth cohort (age group), NOT per-name rates.

    Args:
        names: First names to include
        years: Explicit list of years (overrides data_start_year/data_end_year)
        data_start_year: Start year (must be >= 1985 for name data)
        data_end_year: End year (must be <= 2024 for name data)

    Returns:
        DataFrame with columns:
        birth_year, name, frequency_pct, birth_count, total_male_births,
        education_tertiary_pct, cohort_id

    Raises:
        ValueError: if requested years are outside the 1985-2024 range
        RuntimeError: if API calls fail
    """
    if years is None:
        years = list(range(data_start_year, data_end_year + 1))

    # Validate year range
    years_before_1985 = [y for y in years if y < NAVN_BARO_START_YEAR]
    if years_before_1985:
        raise ValueError(
            f"Name frequency data for years {years_before_1985} is NOT available via the "
            f"DST NavneBarometer API. The API covers {NAVN_BARO_START_YEAR}–{NAVN_BARO_END_YEAR} only. "
            f"Data before 1985 does not exist in any public DST/Statistikbanken endpoint. "
            f"To include pre-1985 cohorts, you would need access to Danmarks Statistik "
            f"microdata (CPR register linkage) via a research agreement."
        )

    years_after_max = [y for y in years if y > NAVN_BARO_END_YEAR]
    if years_after_max:
        raise ValueError(f"Years {years_after_max} exceed NavneBarometer maximum ({NAVN_BARO_END_YEAR})")

    session = requests.Session()

    logger.info(f"Fetching NavneBarometer data for {len(names)} names...")
    raw_name_counts = fetch_name_frequency_births(names, session=session)

    # Build a year → name → count lookup
    year_name_counts: Dict[int, Dict[str, int]] = {}
    for year in years:
        idx = year - NAVN_BARO_START_YEAR
        year_name_counts[year] = {
            name.upper(): raw_name_counts[name.upper()][idx]
            for name in names
        }

    logger.info(f"Fetching FOD total male births for years {min(years)}–{max(years)}...")
    total_births = fetch_total_male_births(years, session=session)

    # Get education rates for each birth cohort
    # We observe the population at age ~35 (a stable mature point post-education)
    # using observation year = birth_year + 35, clamped to HFUDD11 coverage (2008-2024)
    logger.info("Fetching HFUDD11 educational attainment by birth cohort...")
    edu_rates: Dict[int, float] = {}

    # Group years by age_group to minimize API calls
    obs_year_for_birth = {}
    for year in years:
        obs_year = max(2010, min(2024, year + 35))  # observe at ~age 35
        obs_year_for_birth[year] = obs_year

    # Collect unique (age_group, obs_year) pairs
    age_obs_pairs = set()
    for birth_year, obs_year in obs_year_for_birth.items():
        age_grp = birth_year_to_age_group(birth_year, obs_year)
        age_obs_pairs.add((age_grp, obs_year))

    # Fetch education data for each unique (age_group, obs_year)
    edu_cache: Dict[tuple, float] = {}
    for age_grp, obs_yr in sorted(age_obs_pairs):
        rates = fetch_tertiary_education_by_age(age_grp, [obs_yr], session=session)
        edu_cache[(age_grp, obs_yr)] = rates.get(obs_yr, float("nan"))
        logger.info(f"  Birth cohort ~{obs_yr - int(age_grp.split('-')[0])}: "
                    f"age group {age_grp} in {obs_yr} → tertiary edu {edu_cache[(age_grp, obs_yr)]:.1f}%")
        time.sleep(0.2)

    # Build per-birth-year education rate
    for birth_year in years:
        obs_year = obs_year_for_birth[birth_year]
        age_grp = birth_year_to_age_group(birth_year, obs_year)
        edu_rates[birth_year] = edu_cache.get((age_grp, obs_year), float("nan"))

    # Assemble the DataFrame
    rows = []
    for birth_year in years:
        total = total_births.get(birth_year, 0)
        edu_pct = edu_rates.get(birth_year, float("nan"))

        for name in names:
            name_key = name.upper()
            count = year_name_counts[birth_year].get(name_key, 0)
            freq_pct = 100.0 * count / total if total > 0 else float("nan")

            row = {
                "birth_year": birth_year,
                "name": name,
                "frequency_pct": round(freq_pct, 4),
                "birth_count": count,
                "total_male_births": total,
                "education_tertiary_pct": round(edu_pct, 2),
                "cohort_id": f"{name}_{birth_year}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Validate
    assert len(df) == len(years) * len(names), f"Unexpected row count: {len(df)}"
    assert df["frequency_pct"].notna().all(), "NaN in frequency_pct — check total_births for missing years"
    assert (df["birth_count"] >= 0).all(), "Negative birth counts"

    logger.info(f"Real registry data assembled: {len(df)} rows ({len(years)} years × {len(names)} names)")
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 60)
    print("AutoResearchClaw — Real Danish Registry Data Fetch")
    print("=" * 60)
    print()

    # Test with the config's requested range (1985-1995, avail range)
    YEARS = list(range(1985, 1996))
    NAMES = ["Simon", "Lars", "Morten", "Thomas", "Martin", "Mikkel"]

    print(f"Fetching real data for years {YEARS[0]}–{YEARS[-1]}, names: {NAMES}")
    print()

    try:
        df = fetch_registry_data(names=NAMES, years=YEARS)
    except ValueError as e:
        print(f"ERROR (value): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        raise

    print("✓ Data fetched successfully!")
    print()
    print("Sample rows (first 12):")
    print(df.head(12).to_string(index=False))
    print()
    print("Summary by name:")
    summary = df.groupby("name").agg(
        total_births=("birth_count", "sum"),
        mean_freq_pct=("frequency_pct", "mean"),
        peak_freq_pct=("frequency_pct", "max"),
        peak_year=("birth_year", lambda x: df.loc[x.index, "birth_year"][
            df.loc[x.index, "frequency_pct"].idxmax()
        ]),
    ).reset_index()
    print(summary.to_string(index=False))
    print()

    # Test that ValueError is raised for pre-1985 years
    print("Testing error handling for years before 1985...")
    try:
        df_bad = fetch_registry_data(names=["Simon"], years=[1980, 1985])
        print("ERROR: should have raised ValueError!")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:120]}...")

    print()
    print("All tests passed.")
