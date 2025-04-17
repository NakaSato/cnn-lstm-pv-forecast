import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import init, Fore, Style

# Import project modules
from data_preprocessing import (load_data, preprocess_data)
from models import HybridCNNLSTM
from train import inverse_transform_first_feature
from debug_utils import check_processed_data, inspect_model_predictions

# Initialize colorama for colored terminal output
init()

def test_data_loading(file_path, data_type='pv'):
    """
    Test the data loading functionality
    
    Args:
        file_path: Path to the data file
        data_type: Type of data ('pv', 'inverter', or 'weather')
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{Fore.CYAN}Testing data loading for {data_type} data: {file_path}{Style.RESET_ALL}")
    
    try:
        df = load_data(file_path, data_type=data_type)
        
        # Check if the dataframe is empty
        if df.empty:
            print(f"{Fore.RED}Error: Loaded dataframe is empty{Style.RESET_ALL}")
            return {
                'success': False,
                'error': 'Empty dataframe',
                'data': None
            }
        
        # Check for expected columns
        expected_columns = []
        if data_type == 'pv' or data_type == 'inverter':
            expected_columns.append('power_output')
            expected_columns.append('timestamp')
        elif data_type == 'weather':
            expected_columns.extend(['irradiance', 'ambient_temp', 'timestamp'])
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"{Fore.YELLOW}Warning: Missing expected columns: {missing_columns}{Style.RESET_ALL}")
            print(f"Available columns: {df.columns.tolist()}")
        
        # Print data summary
        print(f"\nDataFrame shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")
        print(f"\nData types:\n{df.dtypes}")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        print(f"\nNaN counts:\n{nan_counts}")
        
        print(f"\n{Fore.GREEN}✓ Data loading successful{Style.RESET_ALL}")
        
        return {
            'success': True,
            'data': df,
            'missing_columns': missing_columns,
            'shape': df.shape
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error during data loading: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'data': None
        }

def test_preprocessing_pipeline(file_path, data_type='pv', weather_file_path=None):
    """
    Test the complete preprocessing pipeline
    
    Args:
        file_path: Path to the data file
        data_type: Type of data ('pv' or 'inverter')
        weather_file_path: Optional path to weather data file
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{Fore.CYAN}Testing full preprocessing pipeline for {file_path}{Style.RESET_ALL}")
    
    # Create output directory for test artifacts
    test_output_dir = 'test_results'
    os.makedirs(test_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load the raw data first for comparison
        raw_data_result = test_data_loading(file_path, data_type)
        
        if not raw_data_result['success']:
            print(f"{Fore.RED}Cannot proceed with preprocessing due to data loading failure{Style.RESET_ALL}")
            return {
                'success': False,
                'error': 'Data loading failed',
                'details': raw_data_result
            }
        
        # Test the full preprocessing pipeline
        print(f"\n{Fore.CYAN}Running full preprocessing pipeline...{Style.RESET_ALL}")
        sequence_length = 24  # 24 hours for daily pattern
        forecast_horizon = 24  # Predict next day
        
        X, y, scaler, df_processed = preprocess_data(
            file_path, 
            sequence_length=sequence_length, 
            forecast_horizon=forecast_horizon,
            data_type=data_type,
            weather_file_path=weather_file_path
        )
        
        # Check the output shapes and content
        print(f"\nPreprocessed data shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Processed dataframe shape: {df_processed.shape}")
        
        # Check for NaN or Inf values
        nan_check = {
            'X_has_nan': np.isnan(X).any(),
            'y_has_nan': np.isnan(y).any(),
            'X_has_inf': np.isinf(X).any(),
            'y_has_inf': np.isinf(y).any()
        }
        
        print(f"\nNaN check: X has NaN: {nan_check['X_has_nan']}, y has NaN: {nan_check['y_has_nan']}")
        print(f"Inf check: X has Inf: {nan_check['X_has_inf']}, y has Inf: {nan_check['y_has_inf']}")
        
        # Run diagnostics on the processed data
        diagnostics = check_processed_data(X, y, scaler)
        
        # Save a sample of the processed data for inspection
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        sample_X = X[sample_indices]
        sample_y = y[sample_indices]
        
        # Visualize a sample sequence and its target
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Sample Input Sequence (first feature)')
        plt.plot(sample_X[0, :, 0])
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.title('Sample Target Sequence (first feature)')
        plt.plot(sample_y[0, :, 0])
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{test_output_dir}/sample_sequence_{timestamp}.png')
        plt.close()
        
        # Try to inverse transform a sample
        try:
            # First, reshape to match expected dimensions for inverse transform
            x_sample_flat = sample_X[0, :, 0].reshape(-1, 1)
            y_sample_flat = sample_y[0, :, 0].reshape(-1, 1)
            
            # Inverse transform
            x_original = inverse_transform_first_feature(scaler, x_sample_flat)
            y_original = inverse_transform_first_feature(scaler, y_sample_flat)
            
            # Plot the inverse transformed data
            plt.figure(figsize=(10, 6))
            plt.plot(x_original, label='Input (Original Scale)')
            plt.plot(np.arange(len(x_original), len(x_original) + len(y_original)), 
                     y_original, label='Target (Original Scale)')
            plt.title('Sample Sequence After Inverse Transform')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{test_output_dir}/inverse_transform_sample_{timestamp}.png')
            plt.close()
            
            inverse_transform_success = True
        except Exception as e:
            print(f"{Fore.RED}Error during inverse transform: {e}{Style.RESET_ALL}")
            inverse_transform_success = False
        
        # Create a summary report
        report = {
            'success': True,
            'X_shape': X.shape,
            'y_shape': y.shape,
            'processed_df_shape': df_processed.shape,
            'processed_df_columns': df_processed.columns.tolist(),
            'nan_checks': nan_check,
            'data_diagnostics': diagnostics,
            'inverse_transform_success': inverse_transform_success,
            'sample_visualization_path': f'{test_output_dir}/sample_sequence_{timestamp}.png'
        }
        
        if diagnostics['issues']:
            print(f"\n{Fore.YELLOW}Data quality issues found:{Style.RESET_ALL}")
            for issue in diagnostics['issues']:
                print(f"- {issue}")
        
        print(f"\n{Fore.GREEN}✓ Preprocessing pipeline test completed{Style.RESET_ALL}")
        
        return report
        
    except Exception as e:
        print(f"{Fore.RED}Error during preprocessing pipeline test: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }

def test_model_loading(model_path_prefix):
    """
    Test loading of the model components
    
    Args:
        model_path_prefix: Path prefix to the saved model
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{Fore.CYAN}Testing model loading from {model_path_prefix}{Style.RESET_ALL}")
    
    # Check if model files exist
    model_files = {
        'cnn': f"{model_path_prefix}_cnn.keras",
        'lstm_sunny': f"{model_path_prefix}_lstm_sunny.keras",
        'lstm_cloudy': f"{model_path_prefix}_lstm_cloudy.keras"
    }
    
    missing_files = []
    for name, path in model_files.items():
        if not os.path.exists(path):
            # Try with .h5 extension for backward compatibility
            h5_path = path.replace('.keras', '.h5')
            if os.path.exists(h5_path):
                print(f"{Fore.YELLOW}Found {name} model with .h5 extension instead of .keras{Style.RESET_ALL}")
                model_files[name] = h5_path
            else:
                missing_files.append(name)
                print(f"{Fore.RED}File not found: {path}{Style.RESET_ALL}")
    
    if missing_files:
        print(f"{Fore.RED}Cannot load model - missing files: {missing_files}{Style.RESET_ALL}")
        return {
            'success': False,
            'error': f'Missing model files: {missing_files}',
            'model': None
        }
    
    try:
        # Try to create model with reasonable defaults
        sequence_length = 24
        forecast_horizon = 24
        feature_dims = 32  # Common dimension for these models
        
        model = HybridCNNLSTM(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            feature_dims=feature_dims
        )
        
        # Try to load the models
        model.load_models(model_path_prefix)
        
        # Check the loaded models
        model_info = {
            'cnn_input_shape': model.cnn.input_shape,
            'cnn_output_shape': model.cnn.output_shape,
            'lstm_sunny_input_shape': model.lstm_sunny.input_shape,
            'lstm_sunny_output_shape': model.lstm_sunny.output_shape,
            'lstm_cloudy_input_shape': model.lstm_cloudy.input_shape,
            'lstm_cloudy_output_shape': model.lstm_cloudy.output_shape
        }
        
        print(f"\nLoaded model information:")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        print(f"\n{Fore.GREEN}✓ Model loading successful{Style.RESET_ALL}")
        
        return {
            'success': True,
            'model': model,
            'model_info': model_info
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error during model loading: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'model': None
        }

def test_prediction(model_result, data_path):
    """
    Test the prediction pipeline using a loaded model and test data
    
    Args:
        model_result: Result from test_model_loading
        data_path: Path to test data
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{Fore.CYAN}Testing prediction pipeline with {data_path}{Style.RESET_ALL}")
    
    if not model_result['success'] or model_result['model'] is None:
        print(f"{Fore.RED}Cannot test prediction - model loading failed{Style.RESET_ALL}")
        return {
            'success': False,
            'error': 'Model loading failed'
        }
    
    model = model_result['model']
    
    # Create output directory for test artifacts
    test_output_dir = 'test_results'
    os.makedirs(test_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Get sample data for testing
        print(f"\n{Fore.CYAN}Preprocessing test data...{Style.RESET_ALL}")
        
        # Use the model's parameters for preprocessing
        sequence_length = model.sequence_length
        forecast_horizon = model.forecast_horizon
        
        print(f"Using sequence length: {sequence_length}, forecast horizon: {forecast_horizon}")
        
        # Preprocess the test data
        X, y, scaler, df_processed = preprocess_data(
            data_path,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            data_type='inverter'  # Assuming inverter data for testing
        )
        
        if len(X) == 0:
            print(f"{Fore.RED}Error: No valid sequences created from test data{Style.RESET_ALL}")
            return {
                'success': False,
                'error': 'No valid sequences created'
            }
        
        print(f"Test data shapes - X: {X.shape}, y: {y.shape}")
        
        # Check if we need to pad feature dimensions
        feature_dim_model = model.feature_dims
        feature_dim_data = X.shape[2]
        
        if feature_dim_data != feature_dim_model:
            print(f"{Fore.YELLOW}Feature dimension mismatch: model expects {feature_dim_model}, "
                  f"data has {feature_dim_data}. Padding data...{Style.RESET_ALL}")
            
            # Pad the features to match the model's expected dimensions
            padded_X = np.zeros((X.shape[0], X.shape[1], feature_dim_model))
            padded_X[:, :, :feature_dim_data] = X
            X = padded_X
            
            print(f"Padded data shape: {X.shape}")
        
        # Select a few samples for prediction
        n_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_samples = X[sample_indices]
        y_samples = y[sample_indices]
        
        # Make predictions
        print(f"\n{Fore.CYAN}Making predictions with model...{Style.RESET_ALL}")
        predictions = model.predict(X_samples)
        
        # Check prediction shape
        print(f"Prediction shape: {predictions.shape}")
        print(f"Target shape: {y_samples.shape}")
        
        # Ensure prediction and target have same dimensions for comparison
        if len(predictions.shape) > 2:
            pred_power = predictions[:, :, 0]  # First feature is power output
            y_power = y_samples[:, :, 0]
        else:
            pred_power = predictions
            y_power = y_samples
        
        # Inverse transform for evaluation
        pred_flat = pred_power.reshape(-1, 1)
        y_flat = y_power.reshape(-1, 1)
        
        try:
            y_orig = inverse_transform_first_feature(scaler, y_flat)
            pred_orig = inverse_transform_first_feature(scaler, pred_flat)
            
            # Visualize results
            for i in range(len(X_samples)):
                plt.figure(figsize=(12, 6))
                
                # Plot the input sequence
                plt.subplot(1, 2, 1)
                plt.title(f'Input Sequence {i+1}')
                plt.plot(X_samples[i, :, 0])
                plt.grid(True)
                
                # Plot prediction vs actual
                plt.subplot(1, 2, 2)
                plt.title(f'Prediction vs Actual {i+1}')
                
                # Get the corresponding slices from the flattened arrays
                start_idx = i * y_power.shape[1]
                end_idx = start_idx + y_power.shape[1]
                
                plt.plot(y_orig[start_idx:end_idx], label='Actual')
                plt.plot(pred_orig[start_idx:end_idx], label='Predicted')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'{test_output_dir}/prediction_sample_{i}_{timestamp}.png')
                plt.close()
            
            # Create a comprehensive visualization
            inspect_model_predictions(
                y_orig.flatten(),
                pred_orig.flatten(),
                title='Test Predictions vs Actual',
                save_path=f'{test_output_dir}/prediction_analysis_{timestamp}.png'
            )
            
            print(f"\n{Fore.GREEN}✓ Prediction test completed{Style.RESET_ALL}")
            
            return {
                'success': True,
                'prediction_shape': predictions.shape,
                'target_shape': y_samples.shape,
                'visualization_path': f'{test_output_dir}/prediction_analysis_{timestamp}.png'
            }
            
        except Exception as e:
            print(f"{Fore.RED}Error during inverse transform or visualization: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': f'Error during evaluation: {str(e)}',
                'prediction_shape': predictions.shape
            }
            
    except Exception as e:
        print(f"{Fore.RED}Error during prediction test: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }

def test_end_to_end_pipeline(data_path, model_path_prefix, weather_path=None):
    """
    Test the complete end-to-end pipeline from data loading to prediction
    
    Args:
        data_path: Path to test data
        model_path_prefix: Path prefix to the saved model
        weather_path: Optional path to weather data
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{Fore.CYAN}=== Testing end-to-end pipeline ==={Style.RESET_ALL}")
    print(f"Data path: {data_path}")
    print(f"Model path prefix: {model_path_prefix}")
    if weather_path:
        print(f"Weather path: {weather_path}")
    
    results = {}
    
    # 1. Test data loading
    results['data_loading'] = test_data_loading(data_path, data_type='inverter')
    if not results['data_loading']['success']:
        return results
    
    # 2. Test data preprocessing
    results['preprocessing'] = test_preprocessing_pipeline(
        data_path, 
        data_type='inverter',
        weather_file_path=weather_path
    )
    if not results['preprocessing']['success']:
        return results
    
    # 3. Test model loading
    results['model_loading'] = test_model_loading(model_path_prefix)
    if not results['model_loading']['success']:
        return results
    
    # 4. Test prediction
    results['prediction'] = test_prediction(
        results['model_loading'],
        data_path
    )
    
    # 5. Summarize results
    success = all(r.get('success', False) for r in results.values())
    
    if success:
        print(f"\n{Fore.GREEN}=== End-to-end test completed successfully! ==={Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}=== End-to-end test failed. See individual test results for details. ==={Style.RESET_ALL}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test PV forecasting pipeline components')
    parser.add_argument('--data', type=str, help='Path to test data file')
    parser.add_argument('--weather', type=str, help='Path to weather data file')
    parser.add_argument('--model', type=str, help='Path prefix to model files')
    parser.add_argument('--test-type', type=str, choices=['data-loading', 'preprocessing', 'model-loading', 
                                                         'prediction', 'end-to-end'], 
                       default='end-to-end', help='Type of test to run')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.data:
        # Look for data in the data directory
        data_dir = 'data/inverter'
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                args.data = os.path.join(data_dir, files[0])
                print(f"Using default data file: {args.data}")
            else:
                print(f"{Fore.RED}No CSV files found in {data_dir}{Style.RESET_ALL}")
                return
        else:
            print(f"{Fore.RED}Data directory {data_dir} not found{Style.RESET_ALL}")
            return
    
    if not args.model:
        args.model = 'models/hybrid_model'
        print(f"Using default model path: {args.model}")
    
    if not args.weather and args.test_type in ['preprocessing', 'end-to-end']:
        # Look for weather data
        weather_dir = 'data/weather'
        if os.path.exists(weather_dir):
            files = [f for f in os.listdir(weather_dir) if f.endswith('.csv')]
            if files:
                args.weather = os.path.join(weather_dir, files[0])
                print(f"Using default weather file: {args.weather}")
    
    # Run the requested test
    if args.test_type == 'data-loading':
        test_data_loading(args.data, data_type='inverter')
    
    elif args.test_type == 'preprocessing':
        test_preprocessing_pipeline(args.data, data_type='inverter', weather_file_path=args.weather)
    
    elif args.test_type == 'model-loading':
        test_model_loading(args.model)
    
    elif args.test_type == 'prediction':
        model_result = test_model_loading(args.model)
        if model_result['success']:
            test_prediction(model_result, args.data)
    
    elif args.test_type == 'end-to-end':
        test_end_to_end_pipeline(args.data, args.model, args.weather)

if __name__ == "__main__":
    main()
