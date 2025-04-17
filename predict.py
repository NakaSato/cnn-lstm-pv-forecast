import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
from datetime import datetime, timedelta

from models import HybridCNNLSTM
from data_preprocessing import preprocess_data
from evaluation import evaluate_model, print_metrics
from train import inverse_transform_first_feature
from colorama import init, Fore, Back, Style

def pad_features(X, target_feature_dims=32):
    """
    Pad input features to match expected dimensions
    
    Args:
        X: Input data of shape (samples, sequence_length, features)
        target_feature_dims: Target number of features
        
    Returns:
        Padded input data with shape (samples, sequence_length, target_feature_dims)
    """
    if X.shape[-1] == target_feature_dims:
        return X
    
    # If we have too few features, pad with zeros
    if X.shape[-1] < target_feature_dims:
        samples, seq_len, features = X.shape
        padding_width = target_feature_dims - features
        padded_X = np.zeros((samples, seq_len, target_feature_dims))
        padded_X[:, :, :features] = X
        print(f"Padded input features from {features} to {target_feature_dims}")
        return padded_X
    
    # If we have too many features, truncate
    elif X.shape[-1] > target_feature_dims:
        print(f"Truncating input features from {X.shape[-1]} to {target_feature_dims}")
        return X[:, :, :target_feature_dims]
        
    return X

def load_model_and_predict(data_path, model_path_prefix, sequence_length=24, forecast_horizon=7):
    """
    Load trained models and make predictions on new data
    """
    # Load and preprocess data
    X, y, scaler, df_processed = preprocess_data(
        data_path, 
        sequence_length=sequence_length, 
        forecast_horizon=forecast_horizon
    )
    
    # Get the feature dimension from the data
    feature_dims = X.shape[2]
    
    # Check if the model was trained with a different number of features
    # and pad if necessary (CNN model expects 32 features)
    X_padded = pad_features(X, target_feature_dims=32)
    
    # Load models with the correct feature dimensions
    model = HybridCNNLSTM(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_dims=32  # Use the target feature dimensions that match the saved model
    )
    model.load_models(model_path_prefix)
    
    # Make predictions
    predictions = model.predict(X_padded)
    
    # Check if predictions/targets are multi-dimensional
    if len(predictions.shape) > 2:
        print("Multi-dimensional output detected, using first feature (power output) for evaluation")
        # Extract first feature (power output)
        predictions_power = predictions[:, :, 0]
        y_power = y[:, :, 0]
    else:
        predictions_power = predictions
        y_power = y
    
    # Flatten the sequence dimension for inverse transform
    predictions_flat = predictions_power.reshape(-1, 1)
    y_flat = y_power.reshape(-1, 1)
    
    # Use the inverse transform function to handle multi-dimensional data correctly
    # Inverse transform predictions and actual values
    y_inv = inverse_transform_first_feature(scaler, y_flat).flatten()
    predictions_inv = inverse_transform_first_feature(scaler, predictions_flat).flatten()
    
    # Evaluate predictions
    metrics = evaluate_model(y_inv, predictions_inv)
    print("Model Performance:")
    print_metrics(metrics)
    
    # Plot results (sample of first 100 points)
    plt.figure(figsize=(12, 6))
    plt.plot(y_inv[:100], label='Actual')
    plt.plot(predictions_inv[:100], label='Predicted')
    plt.title('PV Power Forecast vs Actual')
    plt.ylabel('Power Output')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig('plots/new_predictions.png')
    plt.show()
    
    return predictions_inv, y_inv, metrics, df_processed

def predict_next_week(data_path, model_path_prefix, sequence_length=24, forecast_horizon=168):
    """
    Predict 7 days (168 hours if hourly data) ahead from the last available data point
    
    Args:
        data_path: Path to the data file
        model_path_prefix: Path prefix to the saved model
        sequence_length: Length of input sequence (default: 24 hours)
        forecast_horizon: Number of hours to forecast ahead (default: 168 hours = 7 days)
        
    Returns:
        DataFrame with forecasted values and dates
    """
    # Load the data first to get the timestamps
    df = pd.read_csv(data_path)
    
    # Handle case sensitivity in column names
    timestamp_col = None
    for col in df.columns:
        if col.lower() == 'timestamp':
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError(f"Could not find 'timestamp' column in {data_path}. Available columns: {df.columns.tolist()}")
    
    # Get the last timestamp in the dataset
    last_timestamp = pd.to_datetime(df[timestamp_col].iloc[-1])
    print(f"Last timestamp in dataset: {last_timestamp}")
    
    # Process data for prediction (with normal forecast horizon to get properly fitted scaler)
    X, _, scaler, df_processed = preprocess_data(
        data_path, 
        sequence_length=sequence_length, 
        forecast_horizon=24  # Use default horizon for preprocessing
    )
    
    # Get only the last sequence for prediction
    last_sequence = X[-1:] 
    
    # Get the feature dimension from the data
    feature_dims = X.shape[2]
    
    print(f"Input sequence shape: {last_sequence.shape}")
    
    # Check if the model was trained with a different number of features
    # and pad if necessary (CNN model expects 32 features)
    last_sequence_padded = pad_features(last_sequence, target_feature_dims=32)
    
    # Create a model instance with a flexible forecast horizon that we'll determine from the model
    model = HybridCNNLSTM(
        sequence_length=sequence_length,
        forecast_horizon=None,  # We'll set this after loading the model
        feature_dims=32  # Use the target feature dimensions that match the saved model
    )
    
    # Try to load the model
    try:
        model.load_models(model_path_prefix)
        print("Successfully loaded model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("You may need to retrain your model with the correct feature dimensions")
        return None
    
    # Determine the actual forecast horizon of the model
    try:
        test_pred = model.predict(last_sequence_padded)
        actual_forecast_horizon = test_pred.shape[1]
        print(f"Detected model's actual forecast horizon: {actual_forecast_horizon}")
        
        # Update model's forecast horizon to match reality
        model.forecast_horizon = actual_forecast_horizon
    except Exception as e:
        print(f"Error detecting model forecast horizon: {e}")
        print("Continuing with default forecast horizon")
        # If we can't determine, use a default value - likely 7 based on error
        actual_forecast_horizon = 7
        model.forecast_horizon = actual_forecast_horizon
    
    # Initialize the forecast array
    power_forecast = np.zeros(forecast_horizon)
    
    # Keep a copy of the input sequence that we'll update with each iteration
    current_sequence = last_sequence_padded.copy()
    
    # Predict in chunks according to the model's actual forecast horizon
    for i in range(0, forecast_horizon, actual_forecast_horizon):
        # Calculate how many steps we need in this iteration
        steps_this_iter = min(actual_forecast_horizon, forecast_horizon - i)
        
        # Make prediction for the next batch
        try:
            pred_batch = model.predict(current_sequence)
            
            # Extract power output feature (first feature)
            if len(pred_batch.shape) > 2:
                pred_power = pred_batch[0, :steps_this_iter, 0]
            else:
                pred_power = pred_batch[0, :steps_this_iter]
            
            # Store the prediction
            power_forecast[i:i+steps_this_iter] = pred_power
            
            # If this is not the last iteration, update the input sequence for the next prediction
            if i + steps_this_iter < forecast_horizon:
                # Shift the sequence by steps_this_iter and append the new predictions
                # This creates a new rolling window for the next prediction
                
                # First, reshape predictions to match feature dimensions
                if len(pred_batch.shape) > 2:
                    # If multi-feature prediction, use the full prediction
                    new_values = pred_batch[0, :steps_this_iter, :]
                else:
                    # If single-feature prediction, expand to match feature dims
                    new_values = np.zeros((steps_this_iter, current_sequence.shape[2]))
                    new_values[:, 0] = pred_power  # Assume power is first feature
                
                # Move the window forward and add new predictions at the end
                current_sequence[0, :-steps_this_iter, :] = current_sequence[0, steps_this_iter:, :]
                
                # Add new predictions at the end - using the correct feature count to avoid shape mismatch
                feature_count = min(new_values.shape[1], current_sequence.shape[2])
                current_sequence[0, -steps_this_iter:, :feature_count] = new_values[:, :feature_count]
        except Exception as e:
            print(f"Error predicting batch at position {i}: {e}")
            # If prediction fails, fill remaining forecasts with the last valid value
            if i > 0:
                power_forecast[i:] = power_forecast[i-1]
            break
    
    # Reshape for inverse transform
    power_forecast_reshaped = power_forecast.reshape(-1, 1)
    
    # Inverse transform to get actual power values
    power_forecast_actual = inverse_transform_first_feature(scaler, power_forecast_reshaped).flatten()
    
    # Ensure no negative values in power forecast
    power_forecast_actual = np.maximum(power_forecast_actual, 0)
    
    # Generate forecast timestamps (7 days ahead from last timestamp)
    forecast_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(forecast_horizon)]
    
    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'forecasted_power': power_forecast_actual
    })
    
    # Plot the forecast
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_df['timestamp'], forecast_df['forecasted_power'])
    plt.title(f'7-Day PV Power Forecast from {last_timestamp.date()}')
    plt.ylabel('Power Output')
    plt.xlabel('Date')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Save the plot
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, 'seven_day_forecast.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # Save forecast to CSV
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, f'forecast_7day_{datetime.now().strftime("%Y%m%d")}.csv')
    forecast_df.to_csv(csv_path, index=False)
    
    print(f"Forecast saved to {csv_path}")
    print(f"Plot saved to {plot_path}")
    
    return forecast_df

def predict_next_day_energy(inverter_files, model_path_prefix, sequence_length=24, forecast_horizon=24):
    """
    Predict the energy production for the next day (24 hours) for each inverter
    
    Args:
        inverter_files: List of paths to inverter data files
        model_path_prefix: Path prefix to the saved model
        sequence_length: Length of input sequence (default: 24 hours)
        forecast_horizon: Number of hours to forecast ahead (default: 24 hours = 1 day)
        
    Returns:
        Dictionary mapping inverter IDs to their forecasted daily energy (kWh)
    """
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime, timedelta
    
    from data_preprocessing import preprocess_data
    from train import inverse_transform_first_feature
    
    energy_forecasts = {}
    daily_profiles = {}
    
    # Determine the next day's date based on the most recent data
    most_recent_timestamp = None
    
    # Find the most recent timestamp from all inverter files
    for file_path in inverter_files:
        try:
            df = pd.read_csv(file_path)
            timestamp_col = None
            for col in df.columns:
                if col.lower() == 'timestamp':
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamps = pd.to_datetime(df[timestamp_col])
                last_timestamp = timestamps.max()
                
                if most_recent_timestamp is None or last_timestamp > most_recent_timestamp:
                    most_recent_timestamp = last_timestamp
        except Exception as e:
            print(f"Error reading timestamps from {file_path}: {e}")
    
    if most_recent_timestamp is None:
        # If we couldn't determine the most recent timestamp, use current date
        print("Could not determine most recent timestamp from data files. Using current date.")
        today = datetime.now().date()
    else:
        # Use the date from the most recent timestamp
        today = most_recent_timestamp.date()
    
    # Set tomorrow's date
    tomorrow = today + timedelta(days=1)
    print(f"Forecasting energy for date: {tomorrow}")
    
    print(f"Predicting next day energy for {len(inverter_files)} inverters...")
    
    # Initialize the model once
    model = HybridCNNLSTM(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_dims=32  # Use the target feature dimensions that match the saved model
    )
    
    # Load the model
    try:
        model.load_models(model_path_prefix)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Process each inverter file
    for file_path in inverter_files:
        inverter_id = os.path.basename(file_path).replace('.csv', '').replace('processed_', '')
        if 'INVERTER' in inverter_id:
            # Extract just the numeric part if possible
            parts = inverter_id.split('_')
            if len(parts) > 1:
                inverter_id = parts[1]  # Just use the number part
        
        print(f"Processing inverter {inverter_id}...")
        
        try:
            # Process data for prediction
            X, _, scaler, df = preprocess_data(
                file_path, 
                sequence_length=sequence_length, 
                forecast_horizon=forecast_horizon
            )
            
            if len(X) == 0:
                print(f"Warning: No valid input sequences for inverter {inverter_id}")
                energy_forecasts[inverter_id] = 0
                daily_profiles[inverter_id] = np.zeros(forecast_horizon)
                continue
            
            # Get only the last sequence for prediction
            last_sequence = X[-1:] 
            
            # Check if the model was trained with a different number of features
            # and pad if necessary (CNN model expects 32 features)
            last_sequence_padded = pad_features(last_sequence, target_feature_dims=32)
            
            # Make prediction
            predictions = model.predict(last_sequence_padded)
            
            # Extract power output predictions
            if len(predictions.shape) > 2:
                power_predictions = predictions[0, :, 0]  # First sample, all timesteps, first feature
            else:
                power_predictions = predictions[0, :]  # First sample, all timesteps
            
            # Inverse transform to get actual power values (kW)
            power_predictions_reshaped = power_predictions.reshape(-1, 1)
            power_values = inverse_transform_first_feature(scaler, power_predictions_reshaped).flatten()
            
            # Ensure no negative values
            power_values = np.maximum(power_values, 0)
            
            # Calculate energy (kWh) by integrating power over time
            # Since we're predicting hourly values, we can just sum the power values
            energy_forecast = np.sum(power_values)
            
            # Store results
            energy_forecasts[inverter_id] = energy_forecast
            daily_profiles[inverter_id] = power_values
            
            print(f"Inverter {inverter_id}: Forecasted energy for {tomorrow} = {energy_forecast:.2f} kWh")
            
        except Exception as e:
            print(f"Error processing inverter {inverter_id}: {e}")
            energy_forecasts[inverter_id] = 0
            daily_profiles[inverter_id] = np.zeros(forecast_horizon)
    
    # Create a DataFrame with the hourly profile for each inverter
    forecast_hours = [datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=h) 
                     for h in range(forecast_horizon)]
    
    profiles_df = pd.DataFrame({
        'timestamp': forecast_hours
    })
    
    # Add each inverter's profile as a column
    for inverter_id, profile in daily_profiles.items():
        if len(profile) == len(forecast_hours):
            profiles_df[f'inverter_{inverter_id}'] = profile
    
    # Save results to CSV
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save daily totals with explicit forecast date
    totals_df = pd.DataFrame({
        'inverter_id': list(energy_forecasts.keys()),
        'forecasted_energy_kwh': list(energy_forecasts.values()),
        'forecast_date': tomorrow
    })
    
    totals_csv_path = os.path.join(output_dir, f'next_day_energy_forecast_{tomorrow.strftime("%Y%m%d")}.csv')
    totals_df.to_csv(totals_csv_path, index=False)
    
    # Save hourly profiles with explicit forecast date
    profiles_csv_path = os.path.join(output_dir, f'next_day_hourly_profile_{tomorrow.strftime("%Y%m%d")}.csv')
    profiles_df.to_csv(profiles_csv_path, index=False)
    
    print(f"Energy forecast for {tomorrow} saved to {totals_csv_path}")
    print(f"Hourly profiles for {tomorrow} saved to {profiles_csv_path}")
    
    # Create a visualization of the forecasts
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        has_valid_profiles = False
        for inverter_id, profile in daily_profiles.items():
            if len(profile) == len(forecast_hours):
                plt.plot(forecast_hours, profile, label=f'Inverter {inverter_id}')
                has_valid_profiles = True
        
        plt.title(f'Power Production Forecast for {tomorrow.strftime("%Y-%m-%d")}')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
        
        # Only add legend if there are valid profiles
        if has_valid_profiles:
            plt.legend()
        else:
            print("Warning: No valid profiles to display in the plot")
            
        plt.xticks(rotation=45)
        
        # Save the plot
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, f'next_day_forecast_{tomorrow.strftime("%Y%m%d")}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Forecast visualization for {tomorrow} saved to {plot_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return {
        'energy_forecasts': energy_forecasts,
        'daily_profiles': daily_profiles,
        'energy_totals_file': totals_csv_path,
        'hourly_profiles_file': profiles_csv_path,
        'forecast_date': tomorrow
    }

def main():
    # Define your data paths
    data_dir = 'data/inverter'
    model_path_prefix = 'models/hybrid_model'
    
    # Find all inverter files
    inverter_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.startswith('processed_') and f.endswith('.csv')]
    
    if not inverter_files:
        print("No processed inverter files found. Please check the data directory.")
        return
    
    # 1. Next-day energy prediction per inverter
    forecast_results = predict_next_day_energy(
        inverter_files,
        model_path_prefix,
        sequence_length=24,
        forecast_horizon=7  # 24 hours for the next day
    )
    
    if forecast_results:
        print("\nSummary of next day energy forecasts:")
        for inverter_id, energy in forecast_results['energy_forecasts'].items():
            print(f"Inverter {inverter_id}: {energy:.2f} kWh")
        
        # Calculate total plant production
        total_energy = sum(forecast_results['energy_forecasts'].values())
        print(f"\nTotal plant production forecast: {total_energy:.2f} kWh")
    
    # 2. You can also keep the 7-day forecast functionality
    # print("\nGenerating 7-day forecast for first inverter...")
    # forecast_df = predict_next_week(
    #     inverter_files[0],
    #     model_path_prefix,
    #     sequence_length=24,
    #     forecast_horizon=168  # 7 days * 24 hours
    # )

if __name__ == "__main__":
    main()
