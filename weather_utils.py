import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_weather_data_v2(file_path):
    """
    Enhanced loader for weather data with comprehensive format detection
    
    Args:
        file_path: Path to the weather CSV file
        
    Returns:
        DataFrame with standardized weather data
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display first few rows for debugging
    print("First few rows of weather data:")
    print(df.head(3))
    
    # Detect and convert time column to datetime
    time_column = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_column = col
            break
    
    if time_column is None:
        time_column = df.columns[0]  # Assume first column is time
    
    # Try various datetime formats
    try:
        df['timestamp'] = pd.to_datetime(df[time_column], errors='coerce')
    except:
        # Try common formats
        formats = ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M', '%Y/%m/%d %H:%M']
        for fmt in formats:
            try:
                df['timestamp'] = pd.to_datetime(df[time_column], format=fmt, errors='coerce')
                if not df['timestamp'].isna().all():
                    break
            except:
                continue
    
    # Check if we managed to parse any dates
    if 'timestamp' not in df.columns or df['timestamp'].isna().all():
        print(f"Warning: Could not parse time column '{time_column}'")
        # Create a dummy timestamp as last resort
        df['timestamp'] = pd.date_range(start='2025-04-04', periods=len(df), freq='T')
    
    # Standardize column names
    std_names = {
        'Irradiance (W/m2)': 'irradiance',
        'Env Temp (Celsius)': 'ambient_temp',
        'Panel Temp (Celsius)': 'panel_temp',
        'Wind Speed (m/s)': 'wind_speed',
        'Env Humidity (Celsius)': 'humidity',
        'hour': 'hour',
        'day_of_week': 'day_of_week',
        'month': 'month',
        'is_weekend': 'is_weekend',
        'hour_sin': 'hour_sin',
        'hour_cos': 'hour_cos',
        'day_of_year': 'day_of_year',
        'declination': 'declination',
        'hour_angle': 'hour_angle',
        'zenith_angle': 'zenith_angle',
        'Heat_Index': 'heat_index',
        'Temp_Diff': 'temp_diff',
        'Wind_X': 'wind_x',
        'Wind_Y': 'wind_y'
    }
    
    # Rename columns where matches exist
    rename_dict = {col: std for col, std in std_names.items() if col in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Check which standard columns exist
    available_features = [col for col in std_names.values() if col in df.columns]
    print(f"Available weather features: {available_features}")
    
    return df

def visualize_weather_data(df):
    """
    Create visualizations of weather data
    
    Args:
        df: DataFrame containing weather data
    """
    if len(df) == 0:
        print("Empty DataFrame, nothing to visualize")
        return
    
    # Set up figure
    plt.figure(figsize=(15, 12))
    
    # Plot irradiance if available
    if 'irradiance' in df.columns:
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['irradiance'], 'r-')
        plt.title('Solar Irradiance')
        plt.ylabel('W/m²')
        plt.grid(True)
    
    # Plot temperatures if available
    if 'ambient_temp' in df.columns and 'panel_temp' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df['timestamp'], df['ambient_temp'], 'b-', label='Ambient')
        plt.plot(df['timestamp'], df['panel_temp'], 'r-', label='Panel')
        plt.title('Temperatures')
        plt.ylabel('°C')
        plt.legend()
        plt.grid(True)
    
    # Plot wind speed if available
    if 'wind_speed' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['wind_speed'], 'g-')
        plt.title('Wind Speed')
        plt.ylabel('m/s')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/weather_data_overview.png')
    plt.close()

def align_pv_weather(pv_df, weather_df, method='nearest'):
    """
    Align PV inverter data with weather data using different methods
    
    Args:
        pv_df: DataFrame containing PV data with timestamp column
        weather_df: DataFrame containing weather data with timestamp column
        method: Alignment method ('nearest', 'interpolate', or 'resample')
        
    Returns:
        DataFrame with aligned data from both sources
    """
    import pandas as pd
    
    # Ensure timestamps are datetime objects
    pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    if method == 'nearest':
        # Use merge_asof for nearest timestamp matching
        merged = pd.merge_asof(
            pv_df.sort_values('timestamp'),
            weather_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1h'),  # Allow up to 1 hour difference
            suffixes=('', '_weather')
        )
    
    elif method == 'interpolate':
        # Set weather data index to timestamp for interpolation
        weather_reindexed = weather_df.set_index('timestamp')
        
        # Create new weather data reindexed to PV timestamps
        new_weather = weather_reindexed.reindex(
            pd.DatetimeIndex(pv_df['timestamp']),
            method='nearest',  # Use nearest values as initial approximation
        )
        
        # Interpolate any missing values - first convert to float columns
        numeric_cols = new_weather.select_dtypes(include=['number']).columns
        new_weather[numeric_cols] = new_weather[numeric_cols].interpolate(method='time')
        
        # Reset index to get timestamp as column again
        new_weather = new_weather.reset_index()
        
        # Merge with PV data on exact timestamps
        merged = pd.merge(
            pv_df,
            new_weather,
            on='timestamp',
            how='inner',
            suffixes=('', '_weather')
        )
    
    elif method == 'resample':
        # Determine target frequency from timestamps
        timestamps = sorted(set(pv_df['timestamp']) | set(weather_df['timestamp']))
        
        # Calculate time differences and find the most common interval
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                         for i in range(len(timestamps)-1)]
            
            # Find most common time difference
            from collections import Counter
            most_common = Counter(time_diffs).most_common(1)[0][0]
            
            # Convert to pandas frequency string
            if most_common < 1:  # Less than a minute
                target_freq = f"{int(most_common*60)}S"  # Seconds
            elif most_common < 60:  # Less than an hour
                target_freq = f"{int(most_common)}T"  # Minutes
            elif most_common < 1440:  # Less than a day
                target_freq = f"{int(most_common/60)}H"  # Hours
            else:
                target_freq = f"{int(most_common/1440)}D"  # Days
        else:
            # Default to hourly if we can't determine frequency
            target_freq = '1H'
        
        # Resample both datasets to common frequency
        pv_resampled = pv_df.set_index('timestamp').resample(target_freq).mean().reset_index()
        weather_resampled = weather_df.set_index('timestamp').resample(target_freq).mean().reset_index()
        
        # Merge the resampled data
        merged = pd.merge(
            pv_resampled, 
            weather_resampled,
            on='timestamp',
            how='inner'
        )
    
    else:
        raise ValueError("Method must be 'nearest', 'interpolate', or 'resample'")
    
    print(f"Alignment complete using {method} method.")
    print(f"Original PV data: {len(pv_df)} rows")
    print(f"Original weather data: {len(weather_df)} rows")
    print(f"Merged result: {len(merged)} rows")
    
    return merged

def extract_advanced_features(df):
    """
    Extract advanced features from weather data
    
    Args:
        df: DataFrame containing weather data
        
    Returns:
        DataFrame with additional derived features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate composite features
    if 'ambient_temp' in df.columns and 'humidity' in df.columns:
        # Check for NaN values and fill them first
        if df['ambient_temp'].isna().any() or df['humidity'].isna().any():
            print("Filling NaN values in ambient_temp and humidity")
            df['ambient_temp'] = df['ambient_temp'].fillna(df['ambient_temp'].mean())
            df['humidity'] = df['humidity'].fillna(df['humidity'].mean())
            
        # Calculate dew point with improved error handling
        try:
            # Ensure humidity is between 0-100%
            df['humidity'] = df['humidity'].clip(0, 100)
            
            # Calculate dew point temperature using Magnus formula
            # Constants for Magnus formula for temperature range -45°C to 60°C
            alpha = 17.27
            beta = 237.7  # °C
            
            # Add numerical stability checks
            # Ensure ambient_temp is within valid range for the formula
            df['ambient_temp'] = df['ambient_temp'].clip(-45, 60)
            
            # Calculate gamma term in the equation with improved numerical stability
            # Add small epsilon to avoid log(0)
            gamma = alpha * df['ambient_temp'] / (beta + df['ambient_temp']) + np.log((df['humidity'] / 100.0).clip(0.001, 1.0))
            
            # Calculate dew point with checks for division by zero
            denominator = alpha - gamma
            # Handle cases where denominator could be very small or zero
            safe_denom = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
            df['dew_point'] = beta * gamma / safe_denom
            
            # Ensure reasonable dew point values (-50°C to 50°C)
            df['dew_point'] = df['dew_point'].clip(-50, 50)
            
            # Calculate heat index (simplified formula) with bounds checking
            df['heat_index'] = 0.5 * (df['ambient_temp'] + 61.0 + ((df['ambient_temp'] - 68.0) * 1.2) + 
                                     (df['humidity'] * 0.094))
            # Clip heat index to reasonable values
            df['heat_index'] = df['heat_index'].clip(-50, 70)
            
        except Exception as e:
            print(f"Error calculating weather derivatives: {e}")
            print("Using alternative method for dew point calculation")
            # Fallback calculation for dew point
            try:
                # Simpler approximation
                df['dew_point'] = df['ambient_temp'] - ((100 - df['humidity']) / 5)
                df['heat_index'] = df['ambient_temp']  # Just use ambient temp as fallback
            except Exception as e2:
                print(f"Fallback calculation also failed: {e2}")
                # If all else fails, just copy ambient temp
                if 'ambient_temp' in df.columns:
                    df['dew_point'] = df['ambient_temp']
                    df['heat_index'] = df['ambient_temp']
                else:
                    # If even ambient_temp is missing, use placeholders
                    print("No temperature data available for dew point calculation")
                    df['dew_point'] = 20.0  # Default moderate temperature
                    df['heat_index'] = 20.0
    
    # Calculate solar position features if timestamp is available
    if 'timestamp' in df.columns:
        try:
            import datetime as dt
            import pytz
            
            # Convert timestamp to datetime if needed
            if not isinstance(df['timestamp'].iloc[0], dt.datetime):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract hour of day
            df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
            
            # Calculate day of year (1-366)
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            
            # Calculate solar elevation and azimuth angles
            # This is a simplified formula - for more accuracy you'd need pyephem or similar
            lat = 40.0  # Default latitude (Northern Hemisphere mid-latitude)
            # Allow custom latitude override if in the data
            if 'latitude' in df.columns:
                lat = df['latitude'].median()
            
            # Convert day of year to radians
            day_angle = 2 * np.pi * (df['day_of_year'] - 1) / 365.0
            
            # Calculate declination angle
            declination = 0.409 * np.sin(day_angle - 1.39)
            df['declination'] = declination
            
            # Calculate hour angle (solar noon is 0)
            hour_angle = (df['hour'] - 12) * np.pi / 12.0
            df['hour_angle'] = hour_angle
            
            # Calculate zenith angle (angle from vertical)
            lat_rad = np.radians(lat)
            zenith_angle = np.arccos(np.sin(lat_rad) * np.sin(declination) + 
                                    np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
            df['zenith_angle'] = np.degrees(zenith_angle)
            
        except Exception as e:
            print(f"Error calculating solar position: {e}")
    
    # Calculate additional features for machine learning
    if 'ambient_temp' in df.columns:
        # Add time lags for temperature if we have a timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            for lag in [1, 3, 6, 12, 24]:  # Hours
                col_name = f'temp_lag_{lag}h'
                df[col_name] = df['ambient_temp'].shift(lag)
            
        # Calculate temperature derivatives (rate of change)
        # First difference approximates the derivative
        df['temp_derivative'] = df['ambient_temp'].diff() / df['ambient_temp'].shift(1)
    
    if 'irradiance' in df.columns:
        # Calculate irradiance moving statistics
        df['irradiance_rolling_mean_3h'] = df['irradiance'].rolling(window=3, min_periods=1).mean()
        df['irradiance_rolling_std_3h'] = df['irradiance'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Fill any remaining NaN values with appropriate defaults
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

# Example usage:
if __name__ == "__main__":
    # This code runs when the script is executed directly
    weather_file = "data/weather-2025-04-04_cleaned.csv"
    
    try:
        # Load and process weather data
        weather_df = load_weather_data_v2(weather_file)
        
        # Extract advanced features
        weather_df = extract_advanced_features(weather_df)
        
        # Visualize the data
        visualize_weather_data(weather_df)
        
        print("Weather data processing complete")
        print(f"Final weather data shape: {weather_df.shape}")
        print(f"Columns: {weather_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Error processing weather data: {e}")
