# COVID-19 Policy and Vaccination Impact Analysis
# Source Package Initialization

from .data_loader import download_all, download_jhu_cases, download_jhu_deaths, download_owid_vaccinations, download_oxcgrt_policy
from .preprocessor import prepare_master_dataset, load_and_transform_jhu, load_owid, load_oxcgrt
from .visualizer import (
    plot_policy_impact,
    plot_vaccination_efficacy,
    plot_lag_analysis,
    plot_stringency_heatmap,
    plot_event_study
)

__version__ = "1.0.0"
__author__ = "mraknar"
