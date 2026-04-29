"""
COVID-19 Data Visualization Module
===================================
Creates academic-quality visualizations for COVID-19 analysis:
1. Policy Impact (Dual Axis): Cases vs Stringency Index
2. Vaccination Efficacy: Scatter plot
3. Lag Analysis: Correlation with time delays
4. Stringency Heatmap: Country × Time intensity
5. Event Study: Pre/Post lockdown analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from typing import Optional, List, Tuple
import warnings

# Configure matplotlib for publication quality
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for countries
COUNTRY_COLORS = {
    'USA': '#E41A1C',  # Red
    'GBR': '#377EB8',  # Blue
    'SWE': '#4DAF4A',  # Green
    'NZL': '#984EA3',  # Purple
    'DEU': '#FF7F00',  # Orange
    'TUR': '#A65628',  # Brown
    'BRA': '#F781BF',  # Pink
}

# Country display names
COUNTRY_NAMES = {
    'USA': 'United States',
    'GBR': 'United Kingdom',
    'SWE': 'Sweden',
    'NZL': 'New Zealand',
    'DEU': 'Germany',
    'TUR': 'Turkey',
    'BRA': 'Brazil',
}


def set_academic_style():
    """Set academic/publication style for plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)


# ============================================================================
# 1. POLICY IMPACT (DUAL AXIS)
# ============================================================================

def plot_policy_impact(
    df: pd.DataFrame,
    country: str = 'USA',
    date_start: str = '2020-03-01',
    date_end: str = '2022-12-31',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create dual-axis plot showing Cases (left) and Stringency Index (right).
    
    Visualizes the relationship between government policy strictness
    and COVID-19 case counts over time.
    
    Args:
        df: Master dataset
        country: Country ISO-3 code
        date_start: Start date (YYYY-MM-DD)
        date_end: End date (YYYY-MM-DD)
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    # Filter data
    mask = (
        (df['Country_ISO3'] == country) &
        (df['Date'] >= pd.to_datetime(date_start)) &
        (df['Date'] <= pd.to_datetime(date_end))
    )
    df_country = df[mask].copy()
    
    country_name = COUNTRY_NAMES.get(country, country)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    
    # Plot new cases (7-day average) on left axis
    color_cases = '#2C3E50'
    ax1.fill_between(
        df_country['Date'],
        df_country['New_Cases_7day_Avg'],
        alpha=0.3,
        color=color_cases,
        label='New Cases (7-day Avg)'
    )
    ax1.plot(
        df_country['Date'],
        df_country['New_Cases_7day_Avg'],
        color=color_cases,
        linewidth=1.5
    )
    
    # Plot Stringency Index on right axis
    color_policy = '#E74C3C'
    stringency_col = 'Stringency_Index' if 'Stringency_Index' in df_country.columns else 'StringencyIndex_Average'
    
    if stringency_col in df_country.columns:
        ax2.plot(
            df_country['Date'],
            df_country[stringency_col],
            color=color_policy,
            linewidth=2.5,
            label='Stringency Index'
        )
        ax2.fill_between(
            df_country['Date'],
            df_country[stringency_col],
            alpha=0.15,
            color=color_policy
        )
    
    # Configure axes
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('New Cases (7-day Average)', color=color_cases, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Stringency Index (0-100)', color=color_policy, fontsize=12, fontweight='bold')
    
    ax1.tick_params(axis='y', labelcolor=color_cases)
    ax2.tick_params(axis='y', labelcolor=color_policy)
    
    # Set Stringency axis limits
    ax2.set_ylim(0, 100)
    
    # Format x-axis dates
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Title and legend
    fig.suptitle(
        f'COVID-19 Policy Impact Analysis: {country_name}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    
    # Add annotation
    ax1.text(
        0.02, 0.98,
        'Note: Higher Stringency Index indicates stricter government policies',
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        style='italic',
        alpha=0.7
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 2. VACCINATION EFFICACY (SCATTER PLOT)
# ============================================================================

def plot_vaccination_efficacy(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create scatter plot: Vaccination Rate vs New Deaths.
    
    Examines whether higher vaccination rates correlate with
    lower COVID-19 mortality.
    
    Args:
        df: Master dataset
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    # Use data from 2021 onwards (when vaccinations were widespread)
    df_vacc = df[df['Date'] >= '2021-01-01'].copy()
    
    # Create monthly aggregates for cleaner visualization
    df_vacc['YearMonth'] = df_vacc['Date'].dt.to_period('M')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vacc_col = 'people_fully_vaccinated_per_hundred'
    death_col = 'New_Deaths_7day_Avg'
    
    if vacc_col not in df_vacc.columns:
        vacc_col = 'people_vaccinated_per_hundred'
    
    for country in df_vacc['Country_ISO3'].unique():
        df_c = df_vacc[df_vacc['Country_ISO3'] == country]
        
        # Sample data points for cleaner plot
        monthly = df_c.groupby('YearMonth').agg({
            vacc_col: 'mean',
            death_col: 'mean'
        }).reset_index()
        
        ax.scatter(
            monthly[vacc_col],
            monthly[death_col],
            c=COUNTRY_COLORS.get(country, 'gray'),
            label=COUNTRY_NAMES.get(country, country),
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Add trend line (all data)
    all_data = df_vacc.dropna(subset=[vacc_col, death_col])
    if len(all_data) > 10:
        z = np.polyfit(all_data[vacc_col], all_data[death_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(all_data[vacc_col].min(), all_data[vacc_col].max(), 100)
        ax.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.8, linewidth=2, label='Trend')
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(all_data[vacc_col].dropna(), all_data[death_col].dropna())
        ax.text(
            0.98, 0.98,
            f'Pearson r = {corr:.3f}\np-value = {p_value:.4f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    ax.set_xlabel('People Fully Vaccinated (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('New Deaths (7-day Average)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Vaccination Efficacy: Vaccination Rate vs COVID-19 Deaths',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 3. LAG ANALYSIS (DELAYED CORRELATION)
# ============================================================================

def plot_lag_analysis(
    df: pd.DataFrame,
    country: str = 'USA',
    max_lag: int = 30,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create bar chart showing correlation between Stringency Index and
    case changes at different time lags (0-30 days).
    
    Determines how many days after policy implementation the effect
    on case counts becomes visible.
    
    Args:
        df: Master dataset
        country: Country ISO-3 code
        max_lag: Maximum lag days to analyze
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    df_country = df[df['Country_ISO3'] == country].copy()
    df_country = df_country.sort_values('Date').reset_index(drop=True)
    
    stringency_col = 'Stringency_Index' if 'Stringency_Index' in df_country.columns else 'StringencyIndex_Average'
    case_col = 'New_Cases_7day_Avg'
    
    # Calculate correlations at different lags
    correlations = []
    p_values = []
    
    for lag in range(max_lag + 1):
        # Shift cases forward (or stringency backward)
        # If lag=7, we compare today's stringency with cases 7 days later
        stringency = df_country[stringency_col].dropna()
        cases_lagged = df_country[case_col].shift(-lag).dropna()
        
        # Align the series
        min_len = min(len(stringency), len(cases_lagged))
        if min_len > 10:
            corr, p_val = stats.pearsonr(
                stringency.iloc[:min_len].values,
                cases_lagged.iloc[:min_len].values
            )
            correlations.append(corr)
            p_values.append(p_val)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    lags = list(range(max_lag + 1))
    colors = ['#27AE60' if c < 0 else '#E74C3C' for c in correlations]
    
    bars = ax.bar(lags, correlations, color=colors, alpha=0.8, edgecolor='white')
    
    # Highlight significant correlations
    for i, (bar, pval) in enumerate(zip(bars, p_values)):
        if pval and pval < 0.05:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # Mark the optimal lag (strongest negative correlation)
    if correlations:
        min_corr_idx = np.nanargmin(correlations)
        min_corr = correlations[min_corr_idx]
        ax.axvline(x=min_corr_idx, color='#2C3E50', linestyle='--', linewidth=2, alpha=0.7)
        ax.annotate(
            f'Optimal Lag: {min_corr_idx} days\n(r = {min_corr:.3f})',
            xy=(min_corr_idx, min_corr),
            xytext=(min_corr_idx + 3, min_corr - 0.1),
            fontsize=11,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax.set_xlabel('Lag (Days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    
    country_name = COUNTRY_NAMES.get(country, country)
    ax.set_title(
        f'Lag Analysis: Policy Effect Delay in {country_name}\n'
        f'(Correlation between Stringency Index and Future Case Counts)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', label='Negative (Higher stringency → Lower cases)'),
        Patch(facecolor='#E74C3C', label='Positive (Higher stringency → Higher cases)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Note
    ax.text(
        0.02, 0.02,
        'Note: Black borders indicate statistically significant correlations (p < 0.05)',
        transform=ax.transAxes,
        fontsize=9,
        style='italic',
        alpha=0.7
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 4. STRINGENCY HEATMAP
# ============================================================================

def plot_stringency_heatmap(
    df: pd.DataFrame,
    date_start: str = '2020-03-01',
    date_end: str = '2022-12-31',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create heatmap showing Stringency Index over time for all countries.
    
    Compares policy strictness across countries and time periods.
    
    Args:
        df: Master dataset
        date_start: Start date
        date_end: End date
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    # Filter date range
    mask = (
        (df['Date'] >= pd.to_datetime(date_start)) &
        (df['Date'] <= pd.to_datetime(date_end))
    )
    df_filtered = df[mask].copy()
    
    stringency_col = 'Stringency_Index' if 'Stringency_Index' in df_filtered.columns else 'StringencyIndex_Average'
    
    # Resample to weekly for cleaner heatmap
    df_filtered['Week'] = df_filtered['Date'].dt.to_period('W').astype(str)
    
    # Pivot table
    pivot = df_filtered.pivot_table(
        values=stringency_col,
        index='Country_ISO3',
        columns='Week',
        aggfunc='mean'
    )
    
    # Rename index with full country names
    pivot.index = [COUNTRY_NAMES.get(c, c) for c in pivot.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        cmap='RdYlGn_r',  # Red = High stringency, Green = Low
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Stringency Index (0-100)', 'shrink': 0.8},
        ax=ax,
        linewidths=0.1,
        linecolor='white'
    )
    
    # Reduce x-axis labels (show every 4th week)
    xticks = ax.get_xticks()
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Show only every 8th label
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 8 != 0:
            label.set_visible(False)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax.set_title(
        'COVID-19 Government Policy Stringency Over Time\n'
        '(Darker red = Stricter policies)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 5. EVENT STUDY (LOCKDOWN ANALYSIS)
# ============================================================================

def plot_event_study(
    df: pd.DataFrame,
    country: str = 'GBR',
    event_date: str = '2020-03-23',  # UK first lockdown
    event_name: str = 'First National Lockdown',
    window_days: int = 60,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create event study plot showing case trends before/after a policy event.
    
    Centers the timeline on the event date (day 0) and compares
    pre-event and post-event trends.
    
    Args:
        df: Master dataset
        country: Country ISO-3 code
        event_date: Date of the policy event (YYYY-MM-DD)
        event_name: Description of the event
        window_days: Days before and after event to show
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    event_date = pd.to_datetime(event_date)
    
    # Filter data
    mask = (
        (df['Country_ISO3'] == country) &
        (df['Date'] >= event_date - pd.Timedelta(days=window_days)) &
        (df['Date'] <= event_date + pd.Timedelta(days=window_days))
    )
    df_event = df[mask].copy()
    
    # Calculate days from event
    df_event['Days_From_Event'] = (df_event['Date'] - event_date).dt.days
    
    country_name = COUNTRY_NAMES.get(country, country)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Separate pre and post data
    df_pre = df_event[df_event['Days_From_Event'] < 0]
    df_post = df_event[df_event['Days_From_Event'] >= 0]
    
    # Plot cases
    ax.plot(
        df_pre['Days_From_Event'],
        df_pre['New_Cases_7day_Avg'],
        color='#E74C3C',
        linewidth=2.5,
        label='Pre-Event'
    )
    ax.plot(
        df_post['Days_From_Event'],
        df_post['New_Cases_7day_Avg'],
        color='#27AE60',
        linewidth=2.5,
        label='Post-Event'
    )
    
    # Fill areas
    ax.fill_between(
        df_pre['Days_From_Event'],
        df_pre['New_Cases_7day_Avg'],
        alpha=0.2,
        color='#E74C3C'
    )
    ax.fill_between(
        df_post['Days_From_Event'],
        df_post['New_Cases_7day_Avg'],
        alpha=0.2,
        color='#27AE60'
    )
    
    # Event line
    ax.axvline(x=0, color='#2C3E50', linewidth=3, linestyle='--', label='Event Day', alpha=0.8)
    
    # Add event annotation
    y_max = df_event['New_Cases_7day_Avg'].max()
    ax.annotate(
        f'{event_name}\n{event_date.strftime("%B %d, %Y")}',
        xy=(0, y_max * 0.9),
        xytext=(15, y_max * 0.9),
        fontsize=11,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', color='black')
    )
    
    # Calculate and display change statistics
    pre_mean = df_pre['New_Cases_7day_Avg'].tail(7).mean() if len(df_pre) >= 7 else df_pre['New_Cases_7day_Avg'].mean()
    post_mean = df_post['New_Cases_7day_Avg'].head(14).mean() if len(df_post) >= 14 else df_post['New_Cases_7day_Avg'].mean()
    
    if pre_mean > 0:
        pct_change = ((post_mean - pre_mean) / pre_mean) * 100
        change_text = f"Change: {pct_change:+.1f}%"
        ax.text(
            0.98, 0.02,
            f'Pre-event avg (7 days): {pre_mean:,.0f}\n'
            f'Post-event avg (14 days): {post_mean:,.0f}\n'
            f'{change_text}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    ax.set_xlabel('Days from Event', fontsize=12, fontweight='bold')
    ax.set_ylabel('New Cases (7-day Average)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Event Study: Impact of {event_name} on COVID-19 Cases\n{country_name}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(-window_days, window_days)
    ax.set_ylim(bottom=0)
    
    # Add grid for day 0
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# MULTI-COUNTRY COMPARISON
# ============================================================================

def plot_country_comparison(
    df: pd.DataFrame,
    metric: str = 'New_Cases_7day_Per_Million',
    date_start: str = '2020-03-01',
    date_end: str = '2022-12-31',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comparison plot for all countries.
    
    Args:
        df: Master dataset
        metric: Column name to plot
        date_start: Start date
        date_end: End date
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    set_academic_style()
    
    # Filter date range
    mask = (
        (df['Date'] >= pd.to_datetime(date_start)) &
        (df['Date'] <= pd.to_datetime(date_end))
    )
    df_filtered = df[mask].copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for country in df_filtered['Country_ISO3'].unique():
        df_c = df_filtered[df_filtered['Country_ISO3'] == country]
        ax.plot(
            df_c['Date'],
            df_c[metric],
            label=COUNTRY_NAMES.get(country, country),
            color=COUNTRY_COLORS.get(country, 'gray'),
            linewidth=2,
            alpha=0.8
        )
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_title(
        f'COVID-19 {metric.replace("_", " ")}: Country Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# PLOT ALL
# ============================================================================

def create_all_visualizations(df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Generate all visualizations and optionally save them.
    
    Args:
        df: Master dataset
        output_dir: Directory to save figures (optional)
        
    Returns:
        Dictionary of figure objects
    """
    figures = {}
    
    save_kwargs = {'save_path': None, 'show': True}
    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations...")
    
    # 1. Policy Impact
    print("\n1. Policy Impact (USA)...")
    save_path = f"{output_dir}/1_policy_impact_usa.png" if output_dir else None
    figures['policy_impact'] = plot_policy_impact(df, country='USA', save_path=save_path, show=False)
    
    # 2. Vaccination Efficacy
    print("2. Vaccination Efficacy...")
    save_path = f"{output_dir}/2_vaccination_efficacy.png" if output_dir else None
    figures['vaccination_efficacy'] = plot_vaccination_efficacy(df, save_path=save_path, show=False)
    
    # 3. Lag Analysis
    print("3. Lag Analysis (USA)...")
    save_path = f"{output_dir}/3_lag_analysis_usa.png" if output_dir else None
    figures['lag_analysis'] = plot_lag_analysis(df, country='USA', save_path=save_path, show=False)
    
    # 4. Stringency Heatmap
    print("4. Stringency Heatmap...")
    save_path = f"{output_dir}/4_stringency_heatmap.png" if output_dir else None
    figures['stringency_heatmap'] = plot_stringency_heatmap(df, save_path=save_path, show=False)
    
    # 5. Event Study (UK First Lockdown)
    print("5. Event Study (UK Lockdown)...")
    save_path = f"{output_dir}/5_event_study_uk.png" if output_dir else None
    figures['event_study'] = plot_event_study(
        df, 
        country='GBR', 
        event_date='2020-03-23',
        event_name='First National Lockdown',
        save_path=save_path,
        show=False
    )
    
    print("\nAll visualizations created successfully.")
    
    return figures


if __name__ == "__main__":
    # Test visualizations with sample data
    from preprocessor import load_master_dataset
    
    df = load_master_dataset()
    create_all_visualizations(df, output_dir='../data/processed/figures')
