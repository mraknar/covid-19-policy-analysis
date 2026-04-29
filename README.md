# Global COVID-19 Trends and Policy Impact Analysis

A data-driven analysis of how government **policy stringency**, **vaccination coverage**, and **epidemiological outcomes** interacted across seven countries during the COVID-19 pandemic — built from three independent public data sources.

> Data Science, Ankara Yıldırım Beyazıt University
> Author: **Muhammed Remzi Aknar**

---

## Overview

During the COVID-19 pandemic, governments deployed non-pharmaceutical interventions (lockdowns, curfews, mobility restrictions) that differed substantially in both **timing** and **intensity**. This project quantifies the relationship between those interventions, vaccine roll-out, and observed case / death dynamics.

The analysis fuses three datasets, derives a master longitudinal panel, and produces a series of publication-quality visualisations — including correlation matrices, dual-axis policy/case plots, lag-correlation analysis, a stringency heatmap, and an event study around Turkey's April 2020 weekend curfew.

**Countries analysed:**
United States · United Kingdom · Sweden · New Zealand · Germany · Turkey · Brazil

**Headline findings:**
- Policy stringency alone does **not** guarantee epidemic control.
- Vaccination correlates strongly with **reduced mortality** (less so with reduced transmission).
- **Early, well-timed** interventions outperform prolonged late-stage measures.
- Outcomes for countries with similar stringency levels still diverge — timing, vaccination, and societal compliance matter.

---

## Data sources

| Source | Provides | Format |
|---|---|---|
| **JHU CSSE** | Confirmed cases & deaths (global, daily) | Wide CSV |
| **OWID** | Vaccination metrics (per location, daily) | Tidy CSV |
| **OxCGRT** | Stringency Index + 9 underlying policy indicators | Wide CSV |

The **Stringency Index** (0–100) summarises nine policy indicators (school closures, workplace closures, public-event cancellations, gathering restrictions, public-transport closures, stay-at-home orders, internal-movement restrictions, international-travel controls, public-information campaigns); higher values indicate stricter government measures.

`src/data_loader.py` downloads all three datasets directly from their canonical GitHub mirrors — no manual download required.

---

## Methodology

1. **Data ingestion.** Pull the three sources via `data_loader.download_all()`.
2. **Reshape.** JHU is wide-format (one column per date) — melt it to long format and aggregate provinces to country level.
3. **Standardise.** Map all country names to **ISO-3** codes (`US`, `United States`, `USA` → `USA`; `Türkiye`, `Turkey` → `TUR`; etc.).
4. **Merge.** Join all three sources on `(Country_ISO3, Date)` to produce a single master dataset.
5. **Feature engineering.**
   - Daily new cases / deaths from cumulative counts (clipped at zero to swallow data-correction artefacts)
   - **7-day rolling averages** to smooth weekday reporting noise
   - **Per-million normalisation** using a 2022 population estimate per country
   - Case Fatality Rate (CFR)
6. **Analysis.**
   - Contemporaneous correlation matrix
   - **Lag-correlation** (Stringency Index vs future case counts) at lags 0–30 days
   - **Event study** — case trajectory in a window centred on a specific intervention
   - Cross-country comparison scatter (avg stringency vs deaths/million)

---

## Repository layout

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Downloads JHU, OWID, OxCGRT
│   ├── preprocessor.py       # Reshaping, ISO-3 mapping, master-dataset build, feature engineering
│   └── visualizer.py         # All plotting functions (publication-quality matplotlib + seaborn)
├── notebooks/
│   └── analysis.ipynb        # End-to-end driver — download → preprocess → visualise
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Prerequisites
- Python 3.10+
- ~500 MB free disk space (raw datasets total ~80–100 MB; processed master is ~30 MB)

### 2. Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
The notebook runs end-to-end: it downloads the raw data into `data/raw/`, builds and caches the master dataset in `data/processed/master_dataset.csv`, and renders every figure.

### 4. Or use the Python API directly
```python
from src.data_loader import download_all
from src.preprocessor import prepare_master_dataset
from src.visualizer import plot_policy_impact, plot_event_study

download_all()
df = prepare_master_dataset()

plot_policy_impact(df, country='TUR', date_start='2020-03-01', date_end='2022-12-31')
plot_event_study(df, country='TUR',
                 event_date='2020-04-11',
                 event_name='Weekend Curfew Implementation',
                 window_days=45)
```

---

## What the figures show

| # | Figure | Insight |
|---|---|---|
| 1 | **Correlation matrix** | Strong (cases ↔ deaths) link; weak short-term (stringency ↔ cases); negative (vaccination ↔ deaths). |
| 2 | **Policy impact (dual axis)** | Periods of high stringency in Turkey (early 2020, early 2021) coincide with suppressed case growth; premature relaxations are followed by renewed surges. |
| 3 | **Cross-country comparison** | Countries with early, sustained restrictions saw shorter waves, modulated by vaccination roll-out and compliance. |
| 4 | **Vaccination vs death rate (Turkey)** | Mortality declines as vaccination coverage rises, even while case counts remain elevated — supporting the protective effect against severe outcomes. |
| 5 | **Vaccination vs death rate (all countries)** | The inverse vaccination-mortality relationship persists across countries despite policy/timing differences. |
| 6 | **Lag-correlation heatmap & lines** | Policy-on-cases effect is delayed and country-dependent; some countries show weak positive correlations after several weeks, others show minimal or negative association. |
| 7 | **Event study — Turkey weekend curfew (11 Apr 2020)** | Case counts decline gradually after the intervention, suggesting early targeted restrictions can be effective when implemented promptly. |
| 8 | **Avg stringency vs mortality/million** | Positive association overall, but countries with similar stringency display markedly different outcomes — emphasising timing, vaccination, and compliance. |

---

## Key takeaway

> Policy strictness is necessary but not sufficient. Outcomes are shaped jointly by **how early** measures are imposed, **how high** vaccination coverage climbs, and **how compliant** the population is. Public health policy should integrate epidemiological data with behavioural and temporal context — not optimise stringency in isolation.

---

## References

- **JHU CSSE COVID-19 Data**: <https://github.com/CSSEGISandData/COVID-19>
- **Our World in Data — COVID-19 Vaccinations**: <https://github.com/owid/covid-19-data>
- **Oxford COVID-19 Government Response Tracker (OxCGRT)**: <https://github.com/OxCGRT/covid-policy-dataset>

## License

MIT — see [LICENSE](LICENSE).
