import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from datetime import datetime

# Configure plotting style
sns.set_style("whitegrid")

# --- Configuration and Constants ---
BERLIN_FILE = 'berlin_era5_wind_20241231_20241231.csv'
MUNICH_FILE = 'munich_era5_wind_20241231_20241231.csv'

YEAR_TO_ANALYZE = 2024


def load_and_explore_data(filepath, city_name):
    """Loads a dataset, performs basic exploration, and initial cleaning."""
    print(f"\n--- 1. Data Loading and Exploration: {city_name} ---")

    # 🟢 MODIFICATION: Set the column name expected from the CSV to all-lowercase 'timestamp'
    TIME_COLUMN_NAME = 'timestamp'

    try:
        data = pd.read_csv(filepath)
        print(f"✅ Successfully loaded data from: {filepath}")

    except FileNotFoundError:
        print(f"⚠️ ERROR: File not found at {filepath}. Generating synthetic data instead.")
        data = generate_mock_data(city_name)
        TIME_COLUMN_NAME = 'time'  # Revert to 'time' if using mock data
    except Exception as e:
        print(f"An error occurred loading the file: {e}. Using synthetic data.")
        data = generate_mock_data(city_name)
        TIME_COLUMN_NAME = 'time'  # Revert to 'time' if using mock data

    print(f"Shape: {data.shape}")
    print("\nData Types:")
    print(data.dtypes)

    # Ensure the time column is correctly converted and set as index
    if TIME_COLUMN_NAME in data.columns:
        data[TIME_COLUMN_NAME] = pd.to_datetime(data[TIME_COLUMN_NAME])
        data.set_index(TIME_COLUMN_NAME, inplace=True)
    else:
        print(f"❌ CRITICAL ERROR: '{TIME_COLUMN_NAME}' column not found in data. Cannot proceed with analysis.")
        return None

        # Add a dummy 't2m' (temperature) column if not present, as the analysis requires it.
    if 't2m' not in data.columns:
        print("⚠️ 't2m' (Temperature) column not found. Adding a placeholder column.")
        data['t2m'] = 15 - 10 * np.cos(data.index.dayofyear * 2 * np.pi / 365) + np.random.normal(0, 2, len(data))

    # Handle Missing Values: Drop rows with any NaN for simplicity
    print(f"\nMissing values before handling:\n{data[['u10m', 'v10m', 't2m']].isnull().sum()}")
    data_cleaned = data.dropna(subset=['u10m', 'v10m', 't2m'])
    print(f"Missing values after handling (rows dropped): {data.shape[0] - data_cleaned.shape[0]}")

    # Display Summary Statistics
    print("\nSummary Statistics:")
    print(data_cleaned[['u10m', 'v10m', 't2m']].describe())

    return data_cleaned


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def calculate_wind_metrics(df):
    """Calculates wind speed and direction from u10m and v10m."""

    # Wind Speed (Wspd) = sqrt(u^2 + v^2)
    df['Wspd'] = np.sqrt(df['u10m'] ** 2 + df['v10m'] ** 2)

    # Wind Direction (Wdir) in degrees (0=N, 90=E, 180=S, 270=W)
    # atan2 gives results in radians from -pi to pi.
    # Convert to degrees, then normalize to 0-360, adjusting for meteorological convention.
    df['Wdir_rad'] = np.arctan2(df['u10m'], df['v10m'])
    df['Wdir_deg'] = (np.degrees(df['Wdir_rad']) + 360) % 360  # Normalizes to 0-360

    return df


def get_season(date):
    """Determines the season based on the month (Northern Hemisphere)."""
    month = date.month
    if (month in [12, 1, 2]):
        return 'Winter'
    elif (month in [3, 4, 5]):
        return 'Spring'
    elif (month in [6, 7, 8]):
        return 'Summer'
    else:
        return 'Autumn'


def compute_temporal_aggregations(df):
    """Calculates monthly and seasonal averages."""
    df['Month'] = df.index.month
    df['Season'] = df.index.map(get_season)

    # Monthly Averages
    monthly_avg = df.groupby('Month')[['Wspd', 't2m']].mean()
    monthly_avg.index = pd.to_datetime(monthly_avg.index, format='%m').month_name()

    # Seasonal Averages
    seasonal_avg = df.groupby('Season')[['Wspd', 't2m']].mean().reindex(['Winter', 'Spring', 'Summer', 'Autumn'])

    return monthly_avg, seasonal_avg


def statistical_analysis(df, city_name):
    """Performs statistical analysis like extreme values and diurnal patterns."""

    print(f"\n--- 3. Statistical Analysis: {city_name} ---")

    # Identify days/periods with extreme weather (highest wind speeds)
    max_wspd_hour = df['Wspd'].idxmax()
    max_wspd = df['Wspd'].max()
    print(f"Highest Wind Speed: {max_wspd:.2f} m/s on {max_wspd_hour}")

    # Calculate diurnal (daily) patterns in wind speed
    df['Hour'] = df.index.hour
    diurnal_avg = df.groupby('Hour')['Wspd'].mean()

    print("\nDiurnal Wind Speed Averages (m/s):")
    print(diurnal_avg.round(2))

    return diurnal_avg


def create_visualizations(df_berlin, df_munich, monthly_avg_b, seasonal_avg_b, monthly_avg_m, seasonal_avg_m, diurnal_b,
                          diurnal_m):
    """Creates required visualizations."""

    print("\n--- 4. Generating Visualizations ---")

    # --- Visualization 1: Time series plot of monthly average wind speeds ---
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for combined monthly data
    monthly_comparison = pd.DataFrame({
        'Berlin': monthly_avg_b['Wspd'],
        'Munich': monthly_avg_m['Wspd']
    })

    monthly_comparison.plot(kind='line', marker='o', ax=plt.gca())

    plt.title('Monthly Average Wind Speed Comparison (2024)', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='City')
    plt.tight_layout()
    plt.savefig('monthly_avg_wspd.png')
    plt.close()
    print("✅ Created: monthly_avg_wspd.png")

    # --- Visualization 2: Seasonal comparison bar charts ---
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for combined seasonal data
    seasonal_wspd_comp = pd.DataFrame({
        'Berlin': seasonal_avg_b['Wspd'],
        'Munich': seasonal_avg_m['Wspd']
    })

    seasonal_wspd_comp.plot(kind='bar', rot=0, ax=plt.gca())

    plt.title('Seasonal Average Wind Speed Comparison (2024)', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.legend(title='City')
    plt.tight_layout()
    plt.savefig('seasonal_wspd_bar.png')
    plt.close()
    print("✅ Created: seasonal_wspd_bar.png")

    # --- Visualization 3: Diurnal Wind Speed Comparison ---
    plt.figure(figsize=(10, 6))

    diurnal_comp = pd.DataFrame({
        'Berlin': diurnal_b,
        'Munich': diurnal_m
    })

    diurnal_comp.plot(kind='line', marker='.', ax=plt.gca())

    plt.title('Diurnal (Hourly) Average Wind Speed Pattern (2024)', fontsize=14)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.legend(title='City')
    plt.tight_layout()
    plt.savefig('diurnal_wspd_line.png')
    plt.close()
    print("✅ Created: diurnal_wspd_line.png")


def main():
    """Main function to run the ERA5 weather data analysis lab."""

    # --- 1. Load and Explore the Datasets ---
    df_b = load_and_explore_data(BERLIN_FILE, "Berlin")
    df_m = load_and_explore_data(MUNICH_FILE, "Munich")

    if df_b is None or df_m is None:
        print("Exiting due to critical data loading error.")
        return

    # --- 2. Compute Temporal Aggregations ---
    df_b = calculate_wind_metrics(df_b)
    df_m = calculate_wind_metrics(df_m)

    print("\n--- 2. Temporal Aggregations ---")
    monthly_avg_b, seasonal_avg_b = compute_temporal_aggregations(df_b)
    monthly_avg_m, seasonal_avg_m = compute_temporal_aggregations(df_m)

    print("\nBerlin Monthly Average Wind Speed (m/s):")
    print(monthly_avg_b['Wspd'].sort_index().to_frame().T)
    print("\nBerlin Seasonal Averages (Wspd/Temp):")
    print(seasonal_avg_b)

    print("\nMunich Monthly Average Wind Speed (m/s):")
    print(monthly_avg_m['Wspd'].sort_index().to_frame().T)
    print("\nMunich Seasonal Averages (Wspd/Temp):")
    print(seasonal_avg_m)

    # Seasonal Comparison (brief analysis)
    print("\n--- Seasonal Pattern Comparison ---")

    # Determine the city with higher average wind speed for each season
    seasonal_diff = seasonal_avg_b['Wspd'] - seasonal_avg_m['Wspd']

    print(f"Overall average wind speed in Berlin: {df_b['Wspd'].mean():.2f} m/s")
    print(f"Overall average wind speed in Munich: {df_m['Wspd'].mean():.2f} m/s")
    print(f"Berlin is generally {'windier' if seasonal_diff.mean() > 0 else 'less windy'} than Munich.")

    # --- 3. Statistical Analysis ---
    diurnal_b = statistical_analysis(df_b, "Berlin")
    diurnal_m = statistical_analysis(df_m, "Munich")

    # --- 4. Visualization ---
    create_visualizations(df_b, df_m, monthly_avg_b, seasonal_avg_b, monthly_avg_m, seasonal_avg_m, diurnal_b,
                          diurnal_m)


    print("\n" + "=" * 50)
    print("Analysis Complete")
    print("=" * 50)


if __name__ == '__main__':
    main()
