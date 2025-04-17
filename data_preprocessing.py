import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def load_data(file_path, data_type='pv'):
    """Load PV or inverter data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        data_type: Type of data ('pv', 'inverter', or 'weather')
        
    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(file_path)
    
    # Standard column mappings for different file formats
    power_column_names = ['power_output', 'AC_Power', 'Power(kW)', 'Power', 'PAC', 
                         'ActivePower', 'Output', 'Generation', 'kW', 'Energy']
    
    if data_type == 'inverter' or data_type == 'pv':
        # Try to detect power column automatically
        power_col = None
        for col in power_column_names:
            if col in df.columns:
                power_col = col
                break
        
        # If found, rename to standard name
        if power_col and power_col != 'power_output':
            df = df.rename(columns={power_col: 'power_output'})
        
        # Map other common inverter columns to standard names
        column_mapping = {
            'Timestamp': 'timestamp',
            'Date': 'date',
            'Time': 'time',
            'DC_Power': 'dc_power',
            'Voltage_AC': 'voltage_ac',
            'Current_AC': 'current_ac',
            'Frequency': 'frequency',
            'Efficiency': 'efficiency',
            'Temperature': 'temperature',
            'Status_Flags': 'status_flags'  # Added handling for Status_Flags
        }
        
        # Rename columns if they exist in the dataframe
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                
        # Combine date and time to create timestamp if needed
        if 'timestamp' not in df.columns and 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    elif data_type == 'weather':
        # Map weather columns to standard names
        column_mapping = {
            'Time': 'timestamp',
            'Irradiance (W/m2)': 'irradiance',
            'Env Temp (Celsius)': 'ambient_temp',
            'Panel Temp (Celsius)': 'panel_temp',
            'Wind Speed (m/s)': 'wind_speed',
            'Env Humidity (Celsius)': 'humidity',
            'Heat_Index': 'heat_index',
            'Temp_Diff': 'temp_diff',
            'Wind_X': 'wind_x',
            'Wind_Y': 'wind_y'
        }
        
        # Rename columns if they exist in the dataframe
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Display available columns for debugging
    print(f"Available columns in the dataset: {df.columns.tolist()}")
    
    # If there's no power_output column, try to guess which one it might be
    if 'power_output' not in df.columns and (data_type == 'inverter' or data_type == 'pv'):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Check if any column name contains keywords related to power
        power_keywords = ['pac', 'power', 'output', 'kw', 'generation', 'energy', 'active']
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in power_keywords):
                print(f"Using column '{col}' as power output")
                df = df.rename(columns={col: 'power_output'})
                break
    
    return df

def remove_outliers(df, column='power_output'):
    """Remove outliers using IQR method."""
    # Check if the specified column exists, otherwise use the first numeric column
    if column not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            column = numeric_cols[0]
            print(f"Column '{column}' not found. Using '{column}' instead.")
        else:
            print("No numeric columns found for outlier removal.")
            return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def interpolate_missing_values(df, columns=None):
    """Interpolate missing values.
    
    Args:
        df: DataFrame containing data
        columns: List of columns to interpolate (if None, will interpolate all numeric columns)
        
    Returns:
        DataFrame with interpolated values
    """
    if columns is None:
        # Interpolate all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter only existing columns
        numeric_cols = [col for col in columns if col in df.columns]
        
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Check if timestamp column exists for time-based interpolation
    if 'timestamp' in df_copy.columns:
        try:
            # Ensure timestamp is datetime
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            # Set index and interpolate
            df_copy = df_copy.set_index('timestamp')
            for col in numeric_cols:
                df_copy[col] = df_copy[col].interpolate(method='time')
            # Reset index to restore timestamp as column
            df_copy = df_copy.reset_index()
        except Exception as e:
            print(f"Warning: Time-based interpolation failed ({str(e)}), falling back to linear interpolation")
            # Fall back to linear interpolation
            for col in numeric_cols:
                df_copy[col] = df_copy[col].interpolate(method='linear')
    else:
        # If no timestamp, use linear interpolation
        for col in numeric_cols:
            df_copy[col] = df_copy[col].interpolate(method='linear')
    
    # After interpolation, there might still be NaN values at the beginning or end
    # Fill these with appropriate values (forward fill remaining NaNs at the beginning,
    # backward fill remaining NaNs at the end)
    for col in numeric_cols:
        # Forward fill for NaNs at the beginning (use ffill instead of fillna with method)
        df_copy[col] = df_copy[col].ffill()
        
        # Backward fill for any remaining NaNs (those at the end) (use bfill instead of fillna with method)
        df_copy[col] = df_copy[col].bfill()
        
        # If still any NaNs (possible if entire column is NaN), fill with 0
        # This is a last resort to ensure no NaNs remain
        if df_copy[col].isna().any():
            print(f"Warning: Column {col} still contains NaN values after interpolation. Filling with 0.")
            df_copy[col] = df_copy[col].fillna(0)
    
    return df_copy

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, n_steps_in, n_steps_out):
    """Create sequences for LSTM training.
    
    Creates input sequences of shape (samples, n_steps_in, features)
    and output sequences of shape (samples, n_steps_out, features)
    """
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        # Input sequence
        X.append(data[i:i+n_steps_in])
        # Output sequence - keep all features for each prediction timestep
        y.append(data[i+n_steps_in:i+n_steps_in+n_steps_out])
    
    return np.array(X), np.array(y)

def balance_data(X, weather_labels, target_ratio=None):
    """Balance data for CNN training based on weather conditions.
    
    Args:
        X: Input sequences
        weather_labels: Binary labels (0=cloudy, 1=sunny)
        target_ratio: Optional target ratio of sunny:cloudy days (if None, will balance equally)
    
    Returns:
        Balanced X and weather_labels
    """
    # Print input shape information for debugging
    print(f"Input data shape: {X.shape}")
    print(f"Feature dimension: {X.shape[-1]}")
    
    sunny_indices = np.where(weather_labels == 1)[0]
    cloudy_indices = np.where(weather_labels == 0)[0]
    
    n_sunny = len(sunny_indices)
    n_cloudy = len(cloudy_indices)
    
    print(f"Initial data distribution: {n_sunny} sunny samples, {n_cloudy} cloudy samples")
    
    if target_ratio is None:
        # Balance equally
        target_count = min(n_sunny, n_cloudy)
        if n_sunny > target_count:
            # Downsample sunny days
            selected_sunny = np.random.choice(sunny_indices, target_count, replace=False)
            selected_indices = np.concatenate([selected_sunny, cloudy_indices])
        else:
            # Downsample cloudy days
            selected_cloudy = np.random.choice(cloudy_indices, target_count, replace=False)
            selected_indices = np.concatenate([sunny_indices, selected_cloudy])
    else:
        # Use target ratio
        if n_sunny / n_cloudy > target_ratio:
            # Too many sunny days
            target_sunny = int(n_cloudy * target_ratio)
            selected_sunny = np.random.choice(sunny_indices, target_sunny, replace=False)
            selected_indices = np.concatenate([selected_sunny, cloudy_indices])
        else:
            # Too many cloudy days
            target_cloudy = int(n_sunny / target_ratio)
            selected_cloudy = np.random.choice(cloudy_indices, target_cloudy, replace=False)
            selected_indices = np.concatenate([sunny_indices, selected_cloudy])
    
    # Sort indices to maintain original order
    selected_indices = np.sort(selected_indices)
    
    balanced_X = X[selected_indices]
    balanced_labels = weather_labels[selected_indices]
    
    print(f"Balanced data shape: {balanced_X.shape}")
    print(f"Balanced data distribution: {sum(balanced_labels)} sunny samples, {len(balanced_labels) - sum(balanced_labels)} cloudy samples")
    
    return balanced_X, balanced_labels

def extract_pv_features(df, timestamp_col='timestamp', power_col='power_output'):
    """Extract features from historical PV data without external sensors.
    
    Args:
        df: DataFrame containing PV output data
        timestamp_col: Column name for timestamp
        power_col: Column name for power output
        
    Returns:
        DataFrame with additional features
    """
    # Ensure timestamp is datetime
    if timestamp_col in df.columns:
        df['datetime'] = pd.to_datetime(df[timestamp_col])
    else:
        df['datetime'] = pd.to_datetime(df.index)
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Calculate sunrise/sunset approximation based on time of year
    df['daylight_hours'] = 12 + 3 * np.sin((df['day_of_year'] - 81) / 365 * 2 * np.pi)
    
    # Extract statistical features from the power output
    window_sizes = [3, 6, 12, 24]
    
    for window in window_sizes:
        if len(df) < window:
            continue
            
        df[f'rolling_mean_{window}h'] = df[power_col].rolling(window=window).mean()
        df[f'rolling_std_{window}h'] = df[power_col].rolling(window=window).std()
        df[f'rolling_max_{window}h'] = df[power_col].rolling(window=window).max()
        
        if len(df) >= 24 * window:
            df[f'same_hour_{window}d_ago'] = df[power_col].shift(24 * window)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def extract_inverter_features(df):
    """Extract features specific to inverter data.
    
    Args:
        df: DataFrame containing inverter data with standard column names
        
    Returns:
        DataFrame with additional features
    """
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        df['datetime'] = pd.to_datetime(df.index)
    
    # Extract time features (same as PV data)
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Calculate inverter-specific features
    if 'dc_power' in df.columns and 'power_output' in df.columns:
        # Calculate actual efficiency if not already provided
        if 'efficiency' not in df.columns:
            # Avoid division by zero by adding a small constant
            df['efficiency'] = df['power_output'] / (df['dc_power'] + 1e-8)
            # Clip efficiency to realistic values (0-100%)
            df['efficiency'] = df['efficiency'].clip(0, 1)
        
        # Calculate power ratio
        df['power_ratio'] = df['power_output'] / (df['dc_power'].rolling(window=24).max() + 1e-8)
        # Clip ratio to a reasonable range to avoid extreme values
        df['power_ratio'] = df['power_ratio'].clip(0, 1.5)  # Allow slight over 1 for noisy data
    
    # Performance metrics
    if 'power_output' in df.columns:
        window_sizes = [3, 6, 12, 24]
        for window in window_sizes:
            if len(df) < window:
                continue
                
            df[f'rolling_mean_{window}h'] = df['power_output'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}h'] = df['power_output'].rolling(window=window, min_periods=1).std().fillna(0)
            df[f'rolling_max_{window}h'] = df['power_output'].rolling(window=window, min_periods=1).max()
    
    # Add voltage and current features if available
    if 'voltage_ac' in df.columns:
        df['voltage_deviation'] = df['voltage_ac'] - df['voltage_ac'].rolling(window=24, min_periods=1).mean()
    
    if 'current_ac' in df.columns:
        df['current_deviation'] = df['current_ac'] - df['current_ac'].rolling(window=24, min_periods=1).mean()
        
        # Add power factor approximation if both voltage and current are available
        if 'power_output' in df.columns and 'voltage_ac' in df.columns:
            # Calculated apparent power (S = VI)
            df['apparent_power'] = df['voltage_ac'] * df['current_ac']
            # Add small constant to avoid division by zero
            df['power_factor'] = df['power_output'] / (df['apparent_power'] + 1e-8)
            df['power_factor'] = df['power_factor'].clip(0, 1)  # Clip to realistic values
    
    if 'frequency' in df.columns:
        df['frequency_deviation'] = df['frequency'] - df['frequency'].rolling(window=24, min_periods=1).mean()
    
    # Process status flags if available
    if 'status_flags' in df.columns:
        # Convert to string if not already
        df['status_flags'] = df['status_flags'].astype(str)
        
        # Create binary feature for error conditions (simplified approach)
        error_keywords = ['error', 'fault', 'warning', 'alarm', 'fail']
        df['has_error'] = df['status_flags'].apply(
            lambda x: 1 if any(keyword in str(x).lower() for keyword in error_keywords) else 0
        )
    
    # Fill NaN values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def add_cyclical_time_features(df):
    """Add cyclical time features to capture daily and seasonal patterns.
    
    Args:
        df: DataFrame with datetime column
        
    Returns:
        DataFrame with additional cyclical time features
    """
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def detect_weather_patterns(df, threshold=0.65, column='power_output', smooth_window=3):
    """Detect weather patterns (sunny vs cloudy) from power output patterns alone.
    
    Enhanced heuristic: Uses daily patterns, variability, and peak values to classify days.
    
    Args:
        df: DataFrame containing power output data
        threshold: Threshold for sunny/cloudy classification
        column: Column name for power output
        smooth_window: Window size for smoothing the daily pattern
        
    Returns:
        Array of binary weather labels (0=cloudy, 1=sunny)
    """
    if 'datetime' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = df.index
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'])
    
    df['date'] = df['datetime'].dt.date
    
    normalized = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    df['normalized_power'] = normalized
    
    daily_stats = df.groupby('date').agg({
        'normalized_power': ['mean', 'max', 'std'],
        'hour': ['count']
    })
    daily_stats.columns = ['daily_avg', 'daily_max', 'daily_std', 'data_points']
    
    daily_stats = daily_stats[daily_stats['data_points'] >= 12]
    
    if len(daily_stats) > smooth_window:
        daily_stats['smoothed_avg'] = daily_stats['daily_avg'].rolling(window=smooth_window, min_periods=1).mean()
        daily_stats['smoothed_max'] = daily_stats['daily_max'].rolling(window=smooth_window, min_periods=1).mean()
    else:
        daily_stats['smoothed_avg'] = daily_stats['daily_avg']
        daily_stats['smoothed_max'] = daily_stats['daily_max']
    
    daily_stats['composite_score'] = (0.3 * daily_stats['smoothed_avg'] + 
                                      0.6 * daily_stats['smoothed_max'] + 
                                      0.1 * (1 - daily_stats['daily_std']))
    
    weather_labels = (daily_stats['composite_score'] > threshold).astype(int)
    
    return weather_labels.values

def prepare_hybrid_model_data(X, y, weather_labels):
    """Prepare data for the hybrid CNN-LSTM model.
    
    Args:
        X: Input sequences
        y: Output sequences
        weather_labels: Binary weather labels (0=cloudy, 1=sunny)
        
    Returns:
        Dictionary containing data splits for CNN and LSTMs
    """
    sunny_indices = np.where(weather_labels == 1)[0]
    cloudy_indices = np.where(weather_labels == 0)[0]
    
    X_sunny = X[sunny_indices]
    y_sunny = y[sunny_indices]
    
    X_cloudy = X[cloudy_indices]
    y_cloudy = y[cloudy_indices]
    
    return {
        'X': X,
        'y': y,
        'weather_labels': weather_labels,
        'X_sunny': X_sunny,
        'y_sunny': y_sunny,
        'X_cloudy': X_cloudy,
        'y_cloudy': y_cloudy,
        'sunny_count': len(X_sunny),
        'cloudy_count': len(X_cloudy)
    }

def extract_weather_features(df):
    """Extract and process features from weather data.
    
    Args:
        df: DataFrame containing weather data
        
    Returns:
        DataFrame with processed weather features
    """
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        df['datetime'] = pd.to_datetime(df.index)
    
    # Extract time features if not already present
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['datetime'].dt.month
    if 'day_of_year' not in df.columns:
        df['day_of_year'] = df['datetime'].dt.dayofyear
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add cyclical time features if not already present
    if 'hour_sin' not in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    if 'hour_cos' not in df.columns:
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Calculate additional weather features
    if 'irradiance' in df.columns:
        # Calculate rolling statistics for irradiance
        window_sizes = [3, 6, 12, 24]
        for window in window_sizes:
            if len(df) < window:
                continue
                
            df[f'irradiance_rolling_mean_{window}h'] = df['irradiance'].rolling(window=window).mean()
            df[f'irradiance_rolling_std_{window}h'] = df['irradiance'].rolling(window=window).std()
    
    # Calculate clearness index if irradiance is available (ratio of measured to theoretical max irradiance)
    if 'irradiance' in df.columns and 'zenith_angle' in df.columns:
        # Theoretical maximum irradiance (simplified model)
        solar_constant = 1361  # W/m^2
        df['theoretical_max_irradiance'] = solar_constant * np.cos(np.radians(df['zenith_angle']))
        # Avoid division by zero
        df['theoretical_max_irradiance'] = df['theoretical_max_irradiance'].clip(lower=1)
        df['clearness_index'] = df['irradiance'] / df['theoretical_max_irradiance']
        df['clearness_index'] = df['clearness_index'].clip(0, 1)  # Constrain between 0 and 1
    
    # Calculate derived wind features if not already present
    if 'wind_speed' in df.columns and 'wind_x' not in df.columns and 'wind_y' not in df.columns:
        # Assuming wind_direction column exists (in degrees, meteorological convention)
        if 'wind_direction' in df.columns:
            # Convert from meteorological convention to mathematical
            theta = (270 - df['wind_direction']) % 360
            theta_rad = np.radians(theta)
            df['wind_x'] = df['wind_speed'] * np.cos(theta_rad)
            df['wind_y'] = df['wind_speed'] * np.sin(theta_rad)
    
    # Calculate temperature difference if not already present
    if 'ambient_temp' in df.columns and 'panel_temp' in df.columns and 'temp_diff' not in df.columns:
        df['temp_diff'] = df['panel_temp'] - df['ambient_temp']
    
    # Fill NaN values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def merge_pv_weather_data(pv_df, weather_df):
    """
    Merge PV data with weather data, aligning timestamps.
    
    Args:
        pv_df: DataFrame containing PV data
        weather_df: DataFrame containing weather data
        
    Returns:
        Merged DataFrame
    """
    # Ensure timestamp columns are datetime objects
    if 'timestamp' in pv_df.columns:
        pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'])
    elif isinstance(pv_df.index, pd.DatetimeIndex):
        pv_df['timestamp'] = pv_df.index
    
    if 'timestamp' in weather_df.columns:
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    elif isinstance(weather_df.index, pd.DatetimeIndex):
        weather_df['timestamp'] = weather_df.index
    
    # Check for timestamp availability
    if 'timestamp' not in pv_df.columns:
        raise ValueError("PV data does not contain a timestamp column")
    if 'timestamp' not in weather_df.columns:
        raise ValueError("Weather data does not contain a timestamp column")
    
    print(f"PV data timestamps: {pv_df['timestamp'].min()} to {pv_df['timestamp'].max()}")
    print(f"Weather data timestamps: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    
    # Resample both datasets to ensure consistent frequency if needed
    # For this example, we'll use exact timestamp matching
    
    # Merge dataframes
    print(f"PV data shape before merge: {pv_df.shape}")
    print(f"Weather data shape before merge: {weather_df.shape}")
    
    merged_df = pd.merge_asof(
        pv_df.sort_values('timestamp'),
        weather_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1h'),  # Changed 'H' to 'h' to fix deprecation warning
        suffixes=('', '_weather')
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Remove duplicate columns that might have been created during the merge
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_weather')]
    merged_df.drop(columns=duplicate_cols, inplace=True)
    
    return merged_df

def preprocess_data(file_path, sequence_length=24, forecast_horizon=24, data_type='pv', weather_file_path=None):
    """Complete preprocessing pipeline using historical PV or inverter data.
    
    Args:
        file_path: Path to the CSV file
        sequence_length: Number of time steps for input sequence
        forecast_horizon: Number of time steps to forecast
        data_type: Type of data ('pv' or 'inverter')
        weather_file_path: Optional path to weather data CSV file
        
    Returns:
        Processed data ready for model training
    """
    print(f"Loading data from {file_path}")
    df = load_data(file_path, data_type=data_type)
    
    # Check if power_output column exists
    if 'power_output' not in df.columns:
        # Try to identify the power column from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"'power_output' column not found. Using the first numeric column '{numeric_cols[0]}' as power output.")
            df = df.rename(columns={numeric_cols[0]: 'power_output'})
        else:
            raise ValueError("No suitable column found for power output data. Please check your CSV file format.")
    
    # Remove outliers from power output
    df_clean = remove_outliers(df, column='power_output')
    
    # Identify columns for interpolation
    interpolation_columns = ['power_output']
    # Add other important numeric columns if they exist
    potential_numeric_cols = ['dc_power', 'voltage_ac', 'current_ac', 'frequency', 'efficiency']
    interpolation_columns.extend([col for col in potential_numeric_cols if col in df_clean.columns])
    
    # Interpolate missing values in important columns
    df_clean = interpolate_missing_values(df_clean, columns=interpolation_columns)
    
    # Merge with weather data if provided
    if weather_file_path:
        print(f"Loading weather data from {weather_file_path}")
        weather_df = load_weather_data(weather_file_path)
        print(f"Weather data shape: {weather_df.shape}, columns: {weather_df.columns.tolist()}")
        df_clean = merge_pv_weather_data(df_clean, weather_df)
        print(f"Merged data shape: {df_clean.shape}")
    
    if data_type == 'inverter':
        df_features = extract_inverter_features(df_clean)
    else:
        df_features = extract_pv_features(df_clean)
    
    df_features = add_cyclical_time_features(df_features)
    
    # If weather data was merged, make sure its features are preserved
    weather_features = ['irradiance', 'ambient_temp', 'panel_temp', 'wind_speed', 'humidity',
                       'wind_x', 'wind_y', 'temp_diff', 'heat_index', 'zenith_angle',
                       'declination', 'hour_angle', 'Heat_Index']
    
    # Ensure we have the minimum required columns
    required_columns = ['power_output']
    for col in required_columns:
        if col not in df_features.columns:
            raise ValueError(f"Required column '{col}' not found in processed data.")
    
    # Attempt to detect weather patterns if we have sufficient data
    try:
        weather_labels = detect_weather_patterns(df_features)
    except Exception as e:
        print(f"Warning: Could not detect weather patterns: {e}")
        # Generate random weather labels as fallback
        weather_labels = np.random.randint(0, 2, size=len(df_features) // (sequence_length + forecast_horizon))
        print(f"Using random weather labels for {len(weather_labels)} sequences")

    # Define feature columns based on data type and available weather data
    common_features = ['power_output']
    
    # Add time features if available
    time_features = ['hour_sin', 'hour_cos', 
                    'day_of_week', 'month', 'is_weekend',
                    'day_of_year']
    
    common_features.extend([f for f in time_features if f in df_features.columns])
    
    # Add rolling statistics if available
    rolling_features = ['rolling_mean_24h', 'rolling_std_24h', 'rolling_max_24h']
    common_features.extend([f for f in rolling_features if f in df_features.columns])
    
    if data_type == 'inverter':
        extra_features = ['efficiency', 'power_ratio', 'voltage_ac', 'current_ac', 
                         'frequency', 'voltage_deviation', 'current_deviation', 
                         'frequency_deviation', 'power_factor', 'has_error']
        # Add available extra features
        feature_columns = common_features + [f for f in extra_features if f in df_features.columns]
    else:
        feature_columns = common_features
    
    # Add weather features if available
    feature_columns.extend([f for f in weather_features if f in df_features.columns])
    
    # Filter features that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df_features.columns]
    print(f"Using features: {feature_columns}")
    
    data = df_features[feature_columns].values
    normalized_data, scaler = normalize_data(data)
    
    X, y = create_sequences(normalized_data, sequence_length, forecast_horizon)
    
    return X, y, scaler, df_features

def load_weather_data(file_path):
    """
    Load and preprocess the specific weather data format provided
    
    Args:
        file_path: Path to the weather CSV file
        
    Returns:
        DataFrame with processed weather data
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert the Time column to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['Time'])
    except:
        # If Time column is not in a standard format, try to parse it
        print("Converting custom time format to datetime...")
        df['timestamp'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
    
    # Check if we have any valid timestamps
    if df['timestamp'].isna().all():
        raise ValueError("Could not parse timestamps in weather data")
    
    # Rename columns to match expected names
    column_mapping = {
        'Irradiance (W/m2)': 'irradiance',
        'Env Temp (Celsius)': 'ambient_temp',
        'Panel Temp (Celsius)': 'panel_temp',
        'Wind Speed (m/s)': 'wind_speed',
        'Env Humidity (Celsius)': 'humidity',
        'Heat_Index': 'heat_index',
        'Temp_Diff': 'temp_diff',
        'Wind_X': 'wind_x',
        'Wind_Y': 'wind_y'
    }
    
    # Apply column name mapping
    df = df.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                           if col in df.columns})
    
    return df

def prepare_cnn_data(X_data, weather_labels):
    """
    Prepare data for CNN training by formatting the input sequences
    and converting weather labels to categorical format.
    
    Args:
        X_data: Input sequences of shape (samples, sequence_length, features)
        weather_labels: Binary weather classification labels (0=cloudy, 1=sunny)
        
    Returns:
        X_cnn: Formatted input data for CNN
        weather_categorical: One-hot encoded weather labels
    """
    import tensorflow as tf
    
    # Check shape information for debugging
    print(f"X_data shape: {X_data.shape}")
    print(f"Weather labels shape: {weather_labels.shape}")
    
    # Format the input sequences for CNN
    # We can use the sequences directly since CNN can handle 3D input
    X_cnn = X_data
    
    # Convert weather labels to categorical (one-hot encoding)
    weather_categorical = tf.keras.utils.to_categorical(weather_labels, num_classes=2)
    
    print(f"Formatted CNN input shape: {X_cnn.shape}")
    print(f"Weather categorical shape: {weather_categorical.shape}")
    
    return X_cnn, weather_categorical