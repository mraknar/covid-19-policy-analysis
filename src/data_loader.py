"""
COVID-19 Data Loader Module
============================
Downloads COVID-19 data from multiple sources:
- JHU CSSE: Cases and Deaths (Wide format)
- OWID: Vaccination data (Tidy format)
- OxCGRT: Government policy data (Stringency Index)
"""

import os
import requests
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data source URLs
URLS = {
    'jhu_cases': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    'jhu_deaths': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
    'owid_vaccinations': 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv',
    'oxcgrt_policy': 'https://github.com/OxCGRT/covid-policy-dataset/raw/main/data/OxCGRT_compact_national_v1.csv',
}

# Output filenames
FILENAMES = {
    'jhu_cases': 'jhu_cases_global.csv',
    'jhu_deaths': 'jhu_deaths_global.csv',
    'owid_vaccinations': 'owid_vaccinations.csv',
    'oxcgrt_policy': 'oxcgrt_policy.csv',
}

def get_raw_data_path() -> Path:
    """Get the path to the raw data directory."""
    # Navigate from src/ to data/raw/
    base_path = Path(__file__).parent.parent / 'data' / 'raw'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def download_file(url: str, filename: str, timeout: int = 60) -> bool:
    """
    Download a file from URL and save to raw data directory.
    
    Args:
        url: Source URL
        filename: Output filename
        timeout: Request timeout in seconds
        
    Returns:
        True if download successful, False otherwise
    """
    output_path = get_raw_data_path() / filename
    
    try:
        logger.info(f"Downloading: {filename}")
        logger.debug(f"URL: {url}")
        
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(f"Saved: {filename} ({file_size:.1f} KB)")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False
    except IOError as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False


def download_jhu_cases() -> bool:
    """Download JHU CSSE confirmed cases data (wide format)."""
    return download_file(URLS['jhu_cases'], FILENAMES['jhu_cases'])


def download_jhu_deaths() -> bool:
    """Download JHU CSSE deaths data (wide format)."""
    return download_file(URLS['jhu_deaths'], FILENAMES['jhu_deaths'])


def download_owid_vaccinations() -> bool:
    """Download Our World in Data vaccination data (tidy format)."""
    return download_file(URLS['owid_vaccinations'], FILENAMES['owid_vaccinations'])


def download_oxcgrt_policy() -> bool:
    """Download Oxford COVID-19 Government Response Tracker data."""
    return download_file(URLS['oxcgrt_policy'], FILENAMES['oxcgrt_policy'])


def download_all() -> dict:
    """
    Download all COVID-19 datasets.
    
    Returns:
        Dictionary with download status for each dataset
    """
    logger.info("=" * 50)
    logger.info("COVID-19 Data Download Started")
    logger.info("=" * 50)
    
    results = {
        'jhu_cases': download_jhu_cases(),
        'jhu_deaths': download_jhu_deaths(),
        'owid_vaccinations': download_owid_vaccinations(),
        'oxcgrt_policy': download_oxcgrt_policy(),
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Download Complete: {success_count}/{total_count} successful")
    logger.info("=" * 50)
    
    return results


def get_file_path(dataset_name: str) -> Optional[Path]:
    """
    Get the file path for a downloaded dataset.
    
    Args:
        dataset_name: One of 'jhu_cases', 'jhu_deaths', 'owid_vaccinations', 'oxcgrt_policy'
        
    Returns:
        Path to the file if exists, None otherwise
    """
    if dataset_name not in FILENAMES:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None
    
    file_path = get_raw_data_path() / FILENAMES[dataset_name]
    
    if file_path.exists():
        return file_path
    else:
        logger.warning(f"File not found: {file_path}")
        return None


if __name__ == "__main__":
    # Run download when script is executed directly
    download_all()
