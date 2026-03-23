"""Economics domain prompt adapter.

Provides domain-specific prompt blocks for empirical economics
experiments (regression analysis, causal inference, panel data).

Supports two modes:
- DGP mode: synthetic data with known treatment effect (default)
- Registry mode: real public API data (triggered by registry/api keywords in topic)
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks

# Keywords that indicate a topic wants real public API data (not synthetic)
_REGISTRY_KEYWORDS = [
    "statistikbanken", "statbank", "dst.dk", "destatis", "bundeswahlleiter",
    "public registry", "public data", "open data api", "administrative data",
    "population register", "census data", "registry data", "real data",
    "registry-based", "api.statbank", "statistisches bundesamt",
    "german brewers", "brauer-bund", "navnestatistik",
]


def _topic_wants_real_data(topic: str) -> bool:
    """Return True if the topic explicitly asks for real API/registry data."""
    t = topic.lower()
    return any(kw in t for kw in _REGISTRY_KEYWORDS)


class EconomicsPromptAdapter(PromptAdapter):
    """Adapter for economics and social science domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        topic = context.get("topic", "") or getattr(domain, "topic", "") or ""
        use_real_data = _topic_wants_real_data(topic)

        if use_real_data:
            dataset_guidance = self._registry_dataset_guidance(topic)
            hints = self._registry_hints(topic)
        else:
            dataset_guidance = domain.dataset_guidance or (
                "Generate synthetic data with known treatment effect (DGP):\n"
                "- Include treatment, outcome, controls, fixed effects\n"
                "- Simulate realistic correlations and confounders\n"
                "- Use numpy.random.seed for reproducibility"
            )
            hints = domain.code_generation_hints or self._default_hints()

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Economics regressions are fast. Focus on:\n"
                "- Multiple specifications (4-6 columns)\n"
                "- Bootstrap SE if needed (100-500 reps)\n"
                "- Cluster-robust SE for panel data"
            ),
            dataset_guidance=dataset_guidance,
            hp_reporting=domain.hp_reporting_guidance or (
                "Report specification details:\n"
                "HYPERPARAMETERS: {'n_obs': ..., 'n_controls': ..., "
                "'true_effect': ..., 'fe_groups': ..., 'cluster_var': ...}"
            ),
            code_generation_hints=hints,
            output_format_guidance=self._output_format(),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is an **{domain.display_name}** experiment.\n"
            f"Paradigm: progressive specification\n\n"
            "Key principles for economics experiments:\n"
            "1. Start simple (OLS), add complexity progressively\n"
            "2. Report each specification as a column in a regression table\n"
            "3. Use robust/clustered standard errors\n"
            "4. Include at least one robustness check\n"
            "5. Use real data from public APIs when the topic specifies a data source\n"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                "Use Hausman test for FE vs RE choice, "
                "F-test for joint significance, "
                "robust/clustered SE for inference."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Economics result analysis:\n"
                "- Compare coefficient estimates across specifications\n"
                "- Check if treatment effect is robust to controls/FE\n"
                "- Report significance levels (*/**/***)\n"
                "- Discuss economic magnitude, not just statistical significance"
            ),
            statistical_test_guidance=(
                "Use Hausman test, robust SE, cluster SE. "
                "Report R², N, F-statistic for each specification."
            ),
        )

    def _default_hints(self) -> str:
        return (
            "Economics code requirements:\n"
            "1. Generate synthetic data with statsmodels/numpy\n"
            "2. Implement progressive specifications:\n"
            "   - Spec 1: Simple OLS (Y ~ treatment)\n"
            "   - Spec 2: OLS + controls (Y ~ treatment + X1 + X2)\n"
            "   - Spec 3: Fixed effects (Y ~ treatment + X + entity FE)\n"
            "   - Spec 4: IV / 2SLS if applicable\n"
            "3. Use robust/clustered standard errors\n"
            "4. Output regression table to results.json\n"
            "5. Use linearmodels for panel FE, statsmodels for OLS/IV\n"
        )

    def _registry_dataset_guidance(self, topic: str) -> str:
        """Guidance for topics that require real public API data."""
        t = topic.lower()

        # Statistikbanken (Denmark)
        if "statistikbanken" in t or "dst.dk" in t or "statbank" in t:
            return (
                "DATA SOURCE: Statistikbanken (Statistics Denmark) — FREE public API, no auth required.\n"
                "Base URL: https://api.statbank.dk/v1/\n\n"
                "KEY ENDPOINTS:\n"
                "  GET /tables?lang=en  — list all tables\n"
                "  GET /tableinfo/{TABLE_ID}?lang=en  — table schema + variables\n"
                "  POST /data  — fetch data (JSON body with table, variables, values)\n\n"
                "USEFUL TABLES FOR DANISH DEMOGRAPHICS:\n"
                "  FOLK1A  — Population by region, sex, age, marital status (quarterly)\n"
                "  FOD     — Live births by year\n"
                "  HFUDD11 — Highest completed education by age cohort\n"
                "  INDKP101 — Income percentiles by year and group\n"
                "  VU02    — Voter turnout by municipality and election\n\n"
                "NAME DATA: Statistikbanken does NOT have name frequency by birth year.\n"
                "  Use navnestatistik: https://www.dst.dk/da/Statistik/emner/befolkning-og-valg/navne\n"
                "  Or construct from FOLK1A cohort birth year + proxy variables.\n\n"
                "EXAMPLE API CALL:\n"
                "  import urllib.request, json\n"
                "  req = urllib.request.Request(\n"
                "      'https://api.statbank.dk/v1/data',\n"
                "      data=json.dumps({'table': 'FOLK1A', 'format': 'JSON',\n"
                "          'variables': [{'code': 'KØN', 'values': ['M']},\n"
                "                        {'code': 'Tid', 'values': ['*']}]}).encode(),\n"
                "      headers={'Content-Type': 'application/json'}\n"
                "  )\n"
                "  result = json.loads(urllib.request.urlopen(req).read())\n\n"
                "CRITICAL: Pull REAL data from the API. Do NOT generate synthetic data.\n"
                "If a specific variable isn't available, raise a clear DataUnavailableError\n"
                "with the table name and variable you tried — do NOT silently fall back to fake data."
            )

        # Destatis / Bundeswahlleiter (Germany)
        if "destatis" in t or "bundeswahlleiter" in t or "brauer-bund" in t:
            return (
                "DATA SOURCE: Multiple German public data sources — all free, no auth.\n\n"
                "ELECTION DATA (Bundeswahlleiter):\n"
                "  Base: https://www.bundeswahlleiter.de/\n"
                "  API/CSV: https://www.bundeswahlleiter.de/bundestagswahlen/2021/ergebnisse.html\n"
                "  Historical CSV: search bundeswahlleiter.de for 'Ergebnisse früherer Bundestagswahlen'\n"
                "  Key endpoint: https://www.bundeswahlleiter.de/dam/jcr/...csv (varies by election year)\n\n"
                "BEER CONSUMPTION (German Brewers Federation):\n"
                "  Deutscher Brauer-Bund publishes annual Statistik PDF/CSV\n"
                "  Also: Statistisches Bundesamt (destatis.de) table 51311-0001\n"
                "  API: https://www-genesis.destatis.de/genesisWS/rest/2020/data/tablefile?\n"
                "       name=51311-0001&area=all&compress=false&transpose=false&format=csv\n\n"
                "HISTORICAL DATA NOTE: Weimar-era (1920-1933) and WWII-era (1933-1945) data\n"
                "must be sourced from digitized historical archives. Check:\n"
                "  - GESIS (gesis.org) for historical German election data\n"
                "  - Wahlen in der Weimarer Republik dataset\n\n"
                "CRITICAL: Pull REAL data. Do NOT generate synthetic time series.\n"
                "If historical data is unavailable via API, explicitly document the gap\n"
                "and use only verified data points — do NOT interpolate or fabricate."
            )

        # Generic registry/public data
        return (
            "DATA SOURCE: Real public API data — the topic specifies a public data source.\n"
            "Use urllib.request to call the API. No authentication required.\n"
            "Pull actual data — do NOT generate synthetic data.\n"
            "If data is unavailable, raise DataUnavailableError with the source URL tried."
        )

    def _registry_hints(self, topic: str) -> str:
        """Code generation hints for registry/API topics."""
        return (
            "REGISTRY/API DATA CODE REQUIREMENTS:\n"
            "1. Call the real public API — the topic specifies the data source URL\n"
            "2. Use urllib.request (stdlib only — no requests/httpx in sandbox)\n"
            "3. Parse JSON or CSV response into pandas DataFrame\n"
            "4. If an API call fails, raise DataUnavailableError — do NOT fall back to synthetic data\n"
            "5. Implement progressive specifications on the REAL data:\n"
            "   - Spec 1: Simple correlation / OLS (Y ~ treatment)\n"
            "   - Spec 2: OLS + controls\n"
            "   - Spec 3: Fixed effects or causal mediation\n"
            "   - Spec 4: Robustness check (different outcome, different cohort)\n"
            "6. Use robust/clustered standard errors (statsmodels)\n"
            "7. Output regression table to results.json with REAL coefficient estimates\n"
            "8. The results.json MUST contain actual data-derived numbers, not placeholders\n\n"
            "FORBIDDEN:\n"
            "- numpy.random to generate fake outcome data\n"
            "- create_synthetic_* functions\n"
            "- hardcoded TRUE_CAUSAL_EFFECT values\n"
            "- Pretending API data is available when it isn't\n\n"
            "AVAILABLE PACKAGES: numpy, pandas, scipy, statsmodels, linearmodels\n"
            "Use urllib.request for HTTP (stdlib — always available in sandbox)"
        )

    def _output_format(self) -> str:
        return (
            "Output regression table to results.json:\n"
            '{"regression_table": {\n'
            '    "spec_1_ols": {"coeff": 0.15, "se": 0.03, "p": 0.001, "n": 5000, "r2": 0.12},\n'
            '    "spec_2_controls": {"coeff": 0.12, "se": 0.02, "p": 0.001, "n": 5000, "r2": 0.25}\n'
            '},\n'
            ' "data_source": "real_api",\n'
            ' "metadata": {"domain": "economics_empirical", "total_runtime_sec": ...}}'
        )
