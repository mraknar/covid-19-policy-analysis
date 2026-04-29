"""
COVID-19 Data Preprocessor Module
==================================
Handles data cleaning, transformation, and merging:
- JHU Wide-to-Long transformation (melt)
- Country name standardization (ISO-3 mapping)
- Multi-source data merging
- Feature engineering (rolling averages, per-million calculations)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# COUNTRY STANDARDIZATION
# ============================================================================

# Target countries for analysis
TARGET_COUNTRIES = ['USA', 'GBR', 'SWE', 'NZL', 'DEU', 'TUR', 'BRA']

# Country name mapping to ISO-3 codes
COUNTRY_TO_ISO3 = {
    # USA variations
    'US': 'USA',
    'United States': 'USA',
    'United States of America': 'USA',
    
    # UK variations
    'United Kingdom': 'GBR',
    'UK': 'GBR',
    'Great Britain': 'GBR',
    
    # Sweden
    'Sweden': 'SWE',
    
    # New Zealand
    'New Zealand': 'NZL',
    
    # Germany
    'Germany': 'DEU',
    'Deutschland': 'DEU',
    
    # Turkey
    'Turkey': 'TUR',
    'Türkiye': 'TUR',
    'Turkiye': 'TUR',
    
    # Brazil
    'Brazil': 'BRA',
    'Brasil': 'BRA',
}

# ISO-3 to display names (for visualization)
ISO3_TO_DISPLAY = {
    'USA': 'United States',
    'GBR': 'United Kingdom',
    'SWE': 'Sweden',
    'NZL': 'New Zealand',
    'DEU': 'Germany',
    'TUR': 'Turkey',
    'BRA': 'Brazil',
}

# Population data (2022 estimates, in millions)
POPULATION_MILLIONS = {
    'USA': 331.9,
    'GBR': 67.5,
    'SWE': 10.4,
    'NZL': 5.1,
    'DEU': 83.2,
    'TUR': 84.8,
    'BRA': 214.3,
}

# ============================================================================
# DATA PATHS
# ============================================================================

def get_raw_data_path() -> Path:
    """Get the path to the raw data directory."""
    return Path(__file__).parent.parent / 'data' / 'raw'


def get_processed_data_path() -> Path:
    """Get the path to the processed data directory."""
    path = Path(__file__).parent.parent / 'data' / 'processed'
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# JHU DATA TRANSFORMATION (Wide to Long)
# ============================================================================

def load_and_transform_jhu(data_type: str = 'cases') -> pd.DataFrame:
    """
    Load JHU data and transform from wide to long format.
    
    Args:
        data_type: 'cases' or 'deaths'
        
    Returns:
        DataFrame in long format with columns: Country_ISO3, Date, value
    """
    filename = f'jhu_{data_type}_global.csv'
    file_path = get_raw_data_path() / filename
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Please download data first using data_loader.download_all()")
    
    logger.info(f"Loading JHU {data_type} data...")
    df = pd.read_csv(file_path)
    
    # Identify date columns (all columns that are not metadata)
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
    date_columns = [col for col in df.columns if col not in id_vars]
    
    # Melt: Wide -> Long format
    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=date_columns,
        var_name='Date',
        value_name=data_type.capitalize()
    )
    
    # Aggregate by country (sum across provinces/states)
    df_agg = df_long.groupby(['Country/Region', 'Date']).agg({
        data_type.capitalize(): 'sum'
    }).reset_index()
    
    # Standardize country names to ISO-3
    df_agg['Country_ISO3'] = df_agg['Country/Region'].map(COUNTRY_TO_ISO3)
    
    # Filter to target countries only
    df_filtered = df_agg[df_agg['Country_ISO3'].isin(TARGET_COUNTRIES)].copy()
    
    # Parse dates
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format='%m/%d/%y')
    
    # Clean up columns
    df_filtered = df_filtered[['Country_ISO3', 'Date', data_type.capitalize()]]
    df_filtered = df_filtered.sort_values(['Country_ISO3', 'Date']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df_filtered)} rows for {df_filtered['Country_ISO3'].nunique()} countries")
    
    return df_filtered


def load_jhu_combined() -> pd.DataFrame:
    """
    Load and combine JHU cases and deaths data.
    
    Returns:
        DataFrame with columns: Country_ISO3, Date, Cases, Deaths
    """
    df_cases = load_and_transform_jhu('cases')
    df_deaths = load_and_transform_jhu('deaths')
    
    # Merge cases and deaths
    df_combined = pd.merge(
        df_cases,
        df_deaths,
        on=['Country_ISO3', 'Date'],
        how='outer'
    )
    
    return df_combined


# ============================================================================
# OWID VACCINATION DATA
# ============================================================================

def load_owid() -> pd.DataFrame:
    """
    Load and process OWID vaccination data.
    
    Returns:
        DataFrame with vaccination metrics for target countries
    """
    file_path = get_raw_data_path() / 'owid_vaccinations.csv'
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Please download data first using data_loader.download_all()")
    
    logger.info("Loading OWID vaccination data...")
    df = pd.read_csv(file_path)
    
    # OWID uses 'location' column with country names
    # Also has 'iso_code' column with ISO-3 codes
    
    # Map country names and filter
    df['Country_ISO3'] = df['iso_code'].apply(lambda x: x if x in TARGET_COUNTRIES else None)
    
    # Also try location mapping for countries without ISO code
    location_mask = df['Country_ISO3'].isna()
    df.loc[location_mask, 'Country_ISO3'] = df.loc[location_mask, 'location'].map(COUNTRY_TO_ISO3)
    
    # Filter to target countries
    df_filtered = df[df['Country_ISO3'].isin(TARGET_COUNTRIES)].copy()
    
    # Parse dates
    df_filtered['Date'] = pd.to_datetime(df_filtered['date'])
    
    # Select relevant columns
    vaccine_cols = [
        'Country_ISO3', 'Date',
        'total_vaccinations',
        'people_vaccinated',
        'people_fully_vaccinated',
        'total_boosters',
        'daily_vaccinations',
        'total_vaccinations_per_hundred',
        'people_vaccinated_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'daily_vaccinations_per_million',
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in vaccine_cols if col in df_filtered.columns]
    df_filtered = df_filtered[available_cols].copy()
    
    df_filtered = df_filtered.sort_values(['Country_ISO3', 'Date']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df_filtered)} vaccination records for {df_filtered['Country_ISO3'].nunique()} countries")
    
    return df_filtered


# ============================================================================
# OXCGRT POLICY DATA
# ============================================================================

def load_oxcgrt() -> pd.DataFrame:
    """
    Load and process Oxford COVID-19 Government Response Tracker data.
    
    Returns:
        DataFrame with policy metrics and Stringency Index for target countries
    """
    file_path = get_raw_data_path() / 'oxcgrt_policy.csv'
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Please download data first using data_loader.download_all()")
    
    logger.info("Loading OxCGRT policy data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # OxCGRT uses 'CountryCode' (ISO-3) and 'CountryName'
    # Try both mappings
    if 'CountryCode' in df.columns:
        df['Country_ISO3'] = df['CountryCode'].apply(lambda x: x if x in TARGET_COUNTRIES else None)
    elif 'country_code' in df.columns:
        df['Country_ISO3'] = df['country_code'].apply(lambda x: x if x in TARGET_COUNTRIES else None)
    
    # Fallback to country name mapping
    if 'CountryName' in df.columns:
        mask = df['Country_ISO3'].isna()
        df.loc[mask, 'Country_ISO3'] = df.loc[mask, 'CountryName'].map(COUNTRY_TO_ISO3)
    elif 'country_name' in df.columns:
        mask = df['Country_ISO3'].isna()
        df.loc[mask, 'Country_ISO3'] = df.loc[mask, 'country_name'].map(COUNTRY_TO_ISO3)
    
    # Filter to target countries
    df_filtered = df[df['Country_ISO3'].isin(TARGET_COUNTRIES)].copy()
    
    # Parse dates - OxCGRT uses YYYYMMDD format in 'Date' column
    date_col = 'Date' if 'Date' in df_filtered.columns else 'date'
    df_filtered['Date'] = pd.to_datetime(df_filtered[date_col].astype(str), format='%Y%m%d')
    
    # Select key policy columns
    policy_cols = [
        'Country_ISO3', 'Date',
        'StringencyIndex_Average',
        'GovernmentResponseIndex_Average',
        'ContainmentHealthIndex_Average',
        'EconomicSupportIndex',
        'C1M_School closing',
        'C2M_Workplace closing',
        'C3M_Cancel public events',
        'C4M_Restrictions on gatherings',
        'C5M_Close public transport',
        'C6M_Stay at home requirements',
        'C7M_Restrictions on internal movement',
        'C8EV_International travel controls',
        'H1_Public information campaigns',
        'H2_Testing policy',
        'H3_Contact tracing',
        'H6M_Facial Coverings',
    ]
    
    # Keep only available columns
    available_cols = [col for col in policy_cols if col in df_filtered.columns]
    
    # Also check for alternative column names (lowercase, underscores)
    for col in policy_cols:
        alt_col = col.lower().replace(' ', '_')
        if alt_col in df_filtered.columns and col not in available_cols:
            df_filtered[col] = df_filtered[alt_col]
            if col not in available_cols:
                available_cols.append(col)
    
    df_filtered = df_filtered[available_cols].copy()
    df_filtered = df_filtered.sort_values(['Country_ISO3', 'Date']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df_filtered)} policy records for {df_filtered['Country_ISO3'].nunique()} countries")
    
    return df_filtered


# ============================================================================
# MASTER DATASET CREATION
# ============================================================================

def prepare_master_dataset(save: bool = True) -> pd.DataFrame:
    """
    Create master dataset by merging all data sources.
    
    Performs:
    1. Load and transform all datasets
    2. Merge on Country_ISO3 and Date
    3. Calculate derived features (daily changes, rolling averages, per-million)
    
    Args:
        save: If True, save the master dataset to processed folder
        
    Returns:
        Master DataFrame with all metrics
    """
    logger.info("=" * 60)
    logger.info("Creating Master Dataset")
    logger.info("=" * 60)
    
    # Load all datasets
    df_jhu = load_jhu_combined()
    df_owid = load_owid()
    df_oxcgrt = load_oxcgrt()
    
    # Merge JHU with OWID
    logger.info("Merging JHU and OWID data...")
    df_master = pd.merge(
        df_jhu,
        df_owid,
        on=['Country_ISO3', 'Date'],
        how='left'
    )
    
    # Merge with OxCGRT
    logger.info("Merging with OxCGRT policy data...")
    df_master = pd.merge(
        df_master,
        df_oxcgrt,
        on=['Country_ISO3', 'Date'],
        how='left'
    )
    
    # Add display country names
    df_master['Country'] = df_master['Country_ISO3'].map(ISO3_TO_DISPLAY)
    
    # Add population
    df_master['Population_Millions'] = df_master['Country_ISO3'].map(POPULATION_MILLIONS)
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    logger.info("Calculating derived features...")
    
    # Sort for proper calculations
    df_master = df_master.sort_values(['Country_ISO3', 'Date']).reset_index(drop=True)
    
    # Calculate daily new cases and deaths (diff from cumulative)
    df_master['New_Cases'] = df_master.groupby('Country_ISO3')['Cases'].diff()
    df_master['New_Deaths'] = df_master.groupby('Country_ISO3')['Deaths'].diff()
    
    # Handle negative values (data corrections)
    df_master['New_Cases'] = df_master['New_Cases'].clip(lower=0)
    df_master['New_Deaths'] = df_master['New_Deaths'].clip(lower=0)
    
    # 7-day rolling averages
    df_master['New_Cases_7day_Avg'] = df_master.groupby('Country_ISO3')['New_Cases'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df_master['New_Deaths_7day_Avg'] = df_master.groupby('Country_ISO3')['New_Deaths'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Per million calculations
    df_master['Cases_Per_Million'] = df_master['Cases'] / df_master['Population_Millions']
    df_master['Deaths_Per_Million'] = df_master['Deaths'] / df_master['Population_Millions']
    df_master['New_Cases_Per_Million'] = df_master['New_Cases'] / df_master['Population_Millions']
    df_master['New_Deaths_Per_Million'] = df_master['New_Deaths'] / df_master['Population_Millions']
    df_master['New_Cases_7day_Per_Million'] = df_master['New_Cases_7day_Avg'] / df_master['Population_Millions']
    df_master['New_Deaths_7day_Per_Million'] = df_master['New_Deaths_7day_Avg'] / df_master['Population_Millions']
    
    # Case Fatality Rate
    df_master['CFR'] = (df_master['Deaths'] / df_master['Cases'] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Rename Stringency column for easier access
    if 'StringencyIndex_Average' in df_master.columns:
        df_master['Stringency_Index'] = df_master['StringencyIndex_Average']
    
    # Final sort
    df_master = df_master.sort_values(['Country_ISO3', 'Date']).reset_index(drop=True)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Master Dataset Summary:")
    logger.info(f"  Total rows: {len(df_master):,}")
    logger.info(f"  Countries: {df_master['Country_ISO3'].nunique()}")
    logger.info(f"  Date range: {df_master['Date'].min().date()} to {df_master['Date'].max().date()}")
    logger.info(f"  Columns: {len(df_master.columns)}")
    logger.info("=" * 60)
    
    # Save to processed folder
    if save:
        output_path = get_processed_data_path() / 'master_dataset.csv'
        df_master.to_csv(output_path, index=False)
        logger.info(f"Saved master dataset to: {output_path}")
    
    return df_master


def load_master_dataset() -> pd.DataFrame:
    """
    Load the pre-processed master dataset.
    
    Returns:
        Master DataFrame
    """
    file_path = get_processed_data_path() / 'master_dataset.csv'
    
    if not file_path.exists():
        logger.warning("Master dataset not found. Creating it now...")
        return prepare_master_dataset(save=True)
    
    df = pd.read_csv(file_path, parse_dates=['Date'])
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_country_data(df: pd.DataFrame, country_iso3: str) -> pd.DataFrame:
    """Get data for a specific country."""
    return df[df['Country_ISO3'] == country_iso3].copy()


def get_date_range(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Filter data by date range."""
    df_filtered = df.copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(end_date)]
    
    return df_filtered


if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    df = prepare_master_dataset(save=True)
    print(f"\nMaster dataset shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
