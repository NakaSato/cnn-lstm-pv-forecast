import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data_preprocessing import preprocess_data, prepare_cnn_data, detect_weather_patterns
from models import HybridCNNLSTM
from evaluation import evaluate_model, print_metrics
from debug_utils import inspect_model_predictions, check_processed_data
from numerical_stability_utils import check_numerical_stability

# Configuration
# RANDOM_SEED = 42
# SEQUENCE_LENGTH = 24  # One day of hourly data
# FORECAST_HORIZON = 7  # Predict next day
# EPOCHS_CNN = 50
# EPOCHS_LSTM = 100
# BATCH_SIZE = 32
# VALIDATION_SPLIT = 0.2

# Configuration for reproducibility next day
RANDOM_SEED = 42
SEQUENCE_LENGTH = 48   # Use two days of history for prediction
FORECAST_HORIZON = 7  # Predict full next day
EPOCHS_CNN = 50
EPOCHS_LSTM = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def inverse_transform_first_feature(scaler, data, feature_idx=0):
    """
    Apply inverse transform to only the first feature (power output) with robust handling of edge cases
    
    Args:
        scaler: The fitted MinMaxScaler
        data: Data to transform, shape (n_samples,) or (n_samples, 1)
        feature_idx: Index of the feature to inverse transform (default: 0)
        
    Returns:
        Inverse-transformed data for the specified feature
    """
    # Handle empty or all-NaN data
    if data is None or len(data) == 0 or np.all(np.isnan(data)):
        print("Warning: All NaN or empty data passed to inverse_transform_first_feature")
        return np.zeros(1) if len(data) == 0 else np.zeros(data.shape[0])
    
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Create a mask for NaN values
    nan_mask = np.isnan(data)
    
    # Replace NaN with 0 temporarily for transformation
    data_clean = np.copy(data)
    data_clean[nan_mask] = 0
    
    if np.any(~np.isfinite(data_clean)):
        print("Warning: Non-finite values found in data (inf/-inf)")
        # Replace inf/-inf with min/max values
        data_clean[data_clean == np.inf] = 1.0  # Max normalized value
        data_clean[data_clean == -np.inf] = 0.0  # Min normalized value
    
    try:
        # Create a dummy array with the same shape that the scaler expects
        dummy = np.zeros((data_clean.shape[0], scaler.scale_.shape[0]))
        
        # Place our data in the correct feature position
        dummy[:, feature_idx:feature_idx+1] = data_clean
        
        # Inverse transform
        dummy_inverse = scaler.inverse_transform(dummy)
        
        # Get the feature we care about
        result = dummy_inverse[:, feature_idx]
        
        # Replace negative values with 0 (PV power can't be negative)
        result = np.maximum(result, 0)
        
        # Put NaN values back where they were
        if np.any(nan_mask):
            result[nan_mask.flatten()] = np.nan
            
        return result
    except Exception as e:
        print(f"Error during inverse transform: {e}")
        # Return zeros as fallback
        return np.zeros(data.shape[0])

def load_multiple_inverters(inverter_files, weather_file_path, sequence_length, forecast_horizon):
    """
    Load and preprocess data from multiple inverters
    
    Args:
        inverter_files: List of file paths to inverter data
        weather_file_path: Path to weather data file
        sequence_length: Length of input sequences
        forecast_horizon: Number of time steps to forecast
        
    Returns:
        X_combined: Combined input data from all inverters
        y_combined: Combined target data from all inverters
        scaler: Fitted scaler used for normalization
        inverter_indices: Dictionary mapping inverter ID to indices in combined data
        df_clean: Cleaned dataframe with all data
    """
    X_combined = []
    y_combined = []
    inverter_indices = {}
    df_combined = None
    scaler = None
    
    print(f"Loading data from {len(inverter_files)} inverters...")
    
    for i, inverter_file in enumerate(inverter_files):
        inverter_id = os.path.basename(inverter_file).replace('.csv', '')
        print(f"Processing {inverter_id}...")
        
        X, y, current_scaler, df_clean = preprocess_data(
            inverter_file, 
            sequence_length=sequence_length, 
            forecast_horizon=forecast_horizon,
            data_type='inverter',
            weather_file_path=weather_file_path
        )
        
        # Store the starting and ending indices for this inverter
        start_idx = len(X_combined)
        X_combined.extend(X)
        y_combined.extend(y)
        end_idx = len(X_combined)
        
        inverter_indices[inverter_id] = (start_idx, end_idx)
        
        # Use the scaler from the first inverter for consistency
        if scaler is None:
            scaler = current_scaler
            df_combined = df_clean
        
    # Convert lists to numpy arrays
    X_combined = np.array(X_combined)
    y_combined = np.array(y_combined)
    
    print(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")
    print(f"Loaded data from {len(inverter_indices)} inverters")
    
    return X_combined, y_combined, scaler, inverter_indices, df_combined

def clean_data_for_training(X, y):
    """
    Clean data to prevent NaN and Inf values during training
    
    Args:
        X: Input data
        y: Target data
        
    Returns:
        Cleaned X and y
    """
    # Check for and report any NaN or Inf values
    nan_count_X = np.isnan(X).sum()
    inf_count_X = np.isinf(X).sum()
    
    nan_count_y = np.isnan(y).sum()
    inf_count_y = np.isinf(y).sum()
    
    data_points = X.size
    total_y_points = y.size
    
    if nan_count_X > 0 or inf_count_X > 0:
        nan_percent = (nan_count_X / data_points) * 100
        inf_percent = (inf_count_X / data_points) * 100
        print(f"Warning: Input data contains {nan_count_X} NaN values ({nan_percent:.2f}%) and {inf_count_X} Inf values ({inf_percent:.2f}%)")
        # Replace NaN with 0 and clip inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    if nan_count_y > 0 or inf_count_y > 0:
        nan_percent = (nan_count_y / total_y_points) * 100
        inf_percent = (inf_count_y / total_y_points) * 100
        print(f"Warning: Target data contains {nan_count_y} NaN values ({nan_percent:.2f}%) and {inf_count_y} Inf values ({inf_percent:.2f}%)")
        # Replace NaN with 0 and clip inf values
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check data ranges for a sample (to avoid memory issues with large datasets)
    sample_size = min(10000, X.shape[0])
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    x_min, x_max = np.min(X_sample), np.max(X_sample)
    y_min, y_max = np.min(y_sample), np.max(y_sample)
    
    print(f"Data sample ranges - X: [{x_min:.4f}, {x_max:.4f}], y: [{y_min:.4f}, {y_max:.4f}]")
    
    # If data has extreme values, clip it
    if x_min < -10 or x_max > 10:
        print("Warning: Input data has extreme values. Clipping to [-10, 10] range")
        X = np.clip(X, -10, 10)
    
    if y_min < -10 or y_max > 10:
        print("Warning: Target data has extreme values. Clipping to [-10, 10] range")
        y = np.clip(y, -10, 10)
    
    return X, y

def evaluate_by_inverter(model, X, y, scaler, inverter_indices, forecast_horizon):
    """
    Evaluate model performance separately for each inverter
    
    Args:
        model: Trained model
        X: Input data
        y: Target data
        scaler: Scaler for inverse transformation
        inverter_indices: Dictionary mapping inverter ID to indices in data
        forecast_horizon: Number of time steps in the forecast
        
    Returns:
        Dictionary of metrics by inverter
    """
    results = {}
    
    for inverter_id, (start_idx, end_idx) in inverter_indices.items():
        print(f"\nEvaluating {inverter_id}...")
        
        # Get data for this inverter
        X_inv = X[start_idx:end_idx]
        y_inv = y[start_idx:end_idx]
        
        # Make predictions
        predictions = model.predict(X_inv)
        
        # Prepare for evaluation (similar to code in main)
        if len(predictions.shape) > 2:
            predictions_power = predictions[:, :, 0]
            y_inv_power = y_inv[:, :, 0]
        else:
            predictions_power = predictions
            y_inv_power = y_inv
        
        # Flatten and inverse transform
        predictions_flat = predictions_power.reshape(-1, 1)
        y_inv_flat = y_inv_power.reshape(-1, 1)
        
        y_inv_actual = inverse_transform_first_feature(scaler, y_inv_flat).flatten()
        predictions_actual = inverse_transform_first_feature(scaler, predictions_flat).flatten()
        
        # Clean up any NaNs
        y_inv_actual = np.nan_to_num(y_inv_actual)
        predictions_actual = np.nan_to_num(predictions_actual)
        
        # Evaluate
        metrics = evaluate_model(y_inv_actual, predictions_actual)
        results[inverter_id] = metrics
        
        print(f"{inverter_id} Performance:")
        print_metrics(metrics)
        
        # Create a plot for this inverter
        plt.figure(figsize=(12, 6))
        plt.plot(y_inv_actual[:100], label='Actual')
        plt.plot(predictions_actual[:100], label='Predicted')
        plt.legend()
        plt.title(f'{inverter_id} PV Power Forecast vs Actual (First 100 points)')
        plt.savefig(f'plots/forecast_results_{inverter_id}.png')
        plt.close()
        
    return results

def generate_weather_labels(df_combined, X_shape):
    """
    Generate weather labels from the combined dataframe
    
    Args:
        df_combined: DataFrame with combined inverter data
        X_shape: Shape of the input data X to match number of labels
        
    Returns:
        Array of binary weather labels (0=cloudy, 1=sunny)
    """
    print("Generating weather labels...")
    if 'power_output' in df_combined.columns:
        try:
            # Use actual data to detect weather patterns
            weather_labels = detect_weather_patterns(df_combined)
            
            # Make sure we have enough labels
            if len(weather_labels) < X_shape[0]:
                print(f"Warning: Not enough weather labels generated ({len(weather_labels)}). Using random labels.")
                weather_labels = np.random.randint(0, 2, size=(X_shape[0]))
            else:
                # Trim any excess labels
                weather_labels = weather_labels[:X_shape[0]]
                
        except Exception as e:
            print(f"Error during weather pattern detection: {e}")
            weather_labels = np.random.randint(0, 2, size=(X_shape[0]))
    else:
        print("No power_output column found for weather pattern detection. Using random labels.")
        weather_labels = np.random.randint(0, 2, size=(X_shape[0]))
    
    # Report distribution
    sunny_count = np.sum(weather_labels)
    cloudy_count = len(weather_labels) - sunny_count
    print(f"Weather label distribution: {sunny_count} sunny ({sunny_count/len(weather_labels)*100:.1f}%), {cloudy_count} cloudy ({cloudy_count/len(weather_labels)*100:.1f}%)")
    
    return weather_labels

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Load and preprocess data
    try:
        # Weather data file
        weather_file = 'data/weather/weather-2025-04-04_cleaned.csv'
        
        # Find all inverter files
        inverter_files = [f for f in os.listdir('data/inverter') if f.startswith('processed_') and f.endswith('.csv')]
        inverter_paths = [os.path.join('data/inverter', f) for f in inverter_files]
        
        if not inverter_paths:
            raise FileNotFoundError("No inverter files found in data directory")
            
        print(f"Found {len(inverter_paths)} inverter files: {inverter_files}")
        
        # Load data from all inverters
        X, y, scaler, inverter_indices, df_combined = load_multiple_inverters(
            inverter_paths,
            weather_file_path=weather_file,
            sequence_length=SEQUENCE_LENGTH,
            forecast_horizon=FORECAST_HORIZON
        )
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        print("Please check your input data format and column names.")
        return
    
    # Clean data to prevent NaN issues
    X, y = clean_data_for_training(X, y)
    
    # Generate weather labels
    weather_labels = generate_weather_labels(df_combined, X.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test, weather_train, weather_test = train_test_split(
        X, y, weather_labels, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Check that the data is suitable for training
    print("Checking data quality...")
    data_check = check_processed_data(X_train, y_train, scaler)
    if data_check['issues']:
        print("Data quality issues detected:")
        for issue in data_check['issues']:
            print(f"- {issue}")
    
    # Get the feature dimension from the data
    feature_dims = X_train.shape[2]
    print(f"Feature dimensions: {feature_dims}")
    
    # Initialize the hybrid model with the correct feature dimensions
    model = HybridCNNLSTM(
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        feature_dims=feature_dims
    )
    
    # Prepare data for CNN training
    print("Preparing data for CNN training...")
    X_cnn, weather_categorical = prepare_cnn_data(X_train, weather_train)
    
    # Train CNN model for weather classification
    print("\nTraining CNN model for weather pattern classification...")
    cnn_history = model.train_cnn(
        X_cnn, 
        weather_categorical, 
        epochs=EPOCHS_CNN, 
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Check CNN accuracy
    cnn_val_accuracy = cnn_history.history.get('val_accuracy', [0])[-1]
    print(f"CNN validation accuracy: {cnn_val_accuracy:.4f}")
    
    # Plot CNN training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history.get('accuracy', []), label='Train')
    plt.plot(cnn_history.history.get('val_accuracy', []), label='Validation')
    plt.title('CNN Weather Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history.get('loss', []), label='Train')
    plt.plot(cnn_history.history.get('val_loss', []), label='Validation')
    plt.title('CNN Weather Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/cnn_training_history.png')
    plt.close()
    
    # Split sunny and cloudy data for LSTM training
    print("\nSplitting data by weather patterns...")
    sunny_indices = np.where(weather_train == 1)[0]
    cloudy_indices = np.where(weather_train == 0)[0]
    
    X_sunny_train = X_train[sunny_indices]
    y_sunny_train = y_train[sunny_indices]
    
    X_cloudy_train = X_train[cloudy_indices]
    y_cloudy_train = y_train[cloudy_indices]
    
    print(f"Sunny training data: {len(X_sunny_train)} samples")
    print(f"Cloudy training data: {len(X_cloudy_train)} samples")
    
    # Train LSTM for sunny days
    print("\nTraining LSTM model for sunny days...")
    lstm_sunny_history = model.train_lstm_sunny(
        X_sunny_train, 
        y_sunny_train, 
        epochs=EPOCHS_LSTM, 
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Train LSTM for cloudy days
    print("\nTraining LSTM model for cloudy days...")
    lstm_cloudy_history = model.train_lstm_cloudy(
        X_cloudy_train, 
        y_cloudy_train, 
        epochs=EPOCHS_LSTM, 
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Plot LSTM training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_sunny_history.history.get('loss', []), label='Sunny Train')
    plt.plot(lstm_sunny_history.history.get('val_loss', []), label='Sunny Val')
    plt.title('LSTM Sunny Days Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_cloudy_history.history.get('loss', []), label='Cloudy Train')
    plt.plot(lstm_cloudy_history.history.get('val_loss', []), label='Cloudy Val')
    plt.title('LSTM Cloudy Days Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/lstm_training_history.png')
    plt.close()
    
    # Save the trained models
    print("\nSaving trained models...")
    model.save_models('models/hybrid_model')
    
    # Check for numerical stability issues
    print("\nChecking for numerical stability issues...")
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    stability_results = check_numerical_stability(model.cnn, X_sample)
    if stability_results['suggestions']:
        print("Numerical stability suggestions:")
        for suggestion in stability_results['suggestions']:
            print(f"- {suggestion}")
    
    # Make predictions on test data using the hybrid model
    print("\nEvaluating model on test data...")
    predictions = model.predict(X_test)
    
    # Handle multi-dimensional outputs (take only power output feature)
    if len(predictions.shape) > 2:
        predictions_power = predictions[:, :, 0]
        y_test_power = y_test[:, :, 0]
    else:
        predictions_power = predictions
        y_test_power = y_test
    
    # Flatten the sequence dimension for inverse transform
    predictions_flat = predictions_power.reshape(-1, 1)
    y_test_flat = y_test_power.reshape(-1, 1)
    
    # Use the inverse transform function to handle multi-dimensional data correctly
    # Inverse transform predictions and actual values
    y_test_actual = inverse_transform_first_feature(scaler, y_test_flat).flatten()
    predictions_actual = inverse_transform_first_feature(scaler, predictions_flat).flatten()
    
    # Evaluate predictions
    metrics = evaluate_model(y_test_actual, predictions_actual)
    print("\nOverall Model Performance:")
    print_metrics(metrics)
    
    # Create and save visualization of predictions
    inspect_model_predictions(
        y_test_actual, 
        predictions_actual,
        title='Hybrid CNN-LSTM PV Power Forecasting', 
        save_path='plots/overall_forecast_results.png'
    )
    
    # Evaluate by inverter
    print("\nEvaluating performance by inverter...")
    # Need to adjust indices since we're working with a test subset
    adjusted_inverter_indices = {}
    test_data_size = len(X_test)
    test_indices = set(range(test_data_size))
    
    # Run the evaluation per-inverter only on test data
    inverter_metrics = evaluate_by_inverter(
        model, X_test, y_test, scaler, 
        {inv_id: (start, end) for inv_id, (start, end) in inverter_indices.items() 
         if start < len(X_test) and end <= len(X_test)},
        FORECAST_HORIZON
    )
    
    # Save metrics by inverter
    metrics_df = pd.DataFrame.from_dict({inv_id: metrics for inv_id, metrics in inverter_metrics.items()})
    metrics_df.to_csv('results/inverter_metrics.csv')
    print("Inverter-specific metrics saved to results/inverter_metrics.csv")
    
    print("\nTraining and evaluation complete!")
    
if __name__ == "__main__":
    main()
