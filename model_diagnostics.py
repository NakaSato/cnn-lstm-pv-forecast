import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style

init()  # Initialize colorama

def diagnose_model_shape_mismatch(model_path, expected_shape=None):
    """
    Diagnose shape mismatches in saved models
    
    Args:
        model_path: Path to saved model file
        expected_shape: Expected input/output shape tuple (in_shape, out_shape)
        
    Returns:
        Dictionary with diagnostic information
    """
    print(f"{Fore.CYAN}Diagnosing model at: {model_path}{Style.RESET_ALL}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Get model input/output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        print(f"Model input shape: {input_shape}")
        print(f"Model output shape: {output_shape}")
        
        # Check if shapes match expected shapes
        if expected_shape:
            exp_in_shape, exp_out_shape = expected_shape
            input_match = input_shape[1:] == exp_in_shape
            output_match = output_shape[1:] == exp_out_shape
            
            if not input_match:
                print(f"{Fore.RED}Input shape mismatch! Expected {exp_in_shape}, got {input_shape[1:]}{Style.RESET_ALL}")
            
            if not output_match:
                print(f"{Fore.RED}Output shape mismatch! Expected {exp_out_shape}, got {output_shape[1:]}{Style.RESET_ALL}")
        
        # Check for specific shape mismatches mentioned in the error
        if len(output_shape) > 2:
            horizon = output_shape[1]
            print(f"Model forecast horizon: {horizon}")
            
            if horizon == 7 and expected_shape and expected_shape[1][0] == 24:
                print(f"{Fore.YELLOW}Warning: Model has a 7-step forecast horizon but code expects 24 steps{Style.RESET_ALL}")
                print("This would cause the 'could not broadcast input array from shape (7,32) into shape (24,32)' error")
                print(f"{Fore.GREEN}Solution: Update the forecast_horizon parameter to 7 when initializing HybridCNNLSTM{Style.RESET_ALL}")
        
        return {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "model_horizon": output_shape[1] if len(output_shape) > 2 else 1,
            "success": True
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error examining model: {str(e)}{Style.RESET_ALL}")
        return {
            "error": str(e),
            "success": False
        }

def get_model_summary(model_path):
    """
    Get a summary of model architecture
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        String with model summary
    """
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Redirect summary output to a string
        import io
        summary_str = io.StringIO()
        model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        
        return summary_str.getvalue()
    except Exception as e:
        return f"Error getting model summary: {str(e)}"

def main():
    # Define model paths
    models_dir = 'models'
    model_files = {
        'cnn': 'hybrid_model_cnn.keras',
        'lstm_sunny': 'hybrid_model_lstm_sunny.keras',
        'lstm_cloudy': 'hybrid_model_lstm_cloudy.keras'
    }
    
    # Check if models exist
    for name, file in model_files.items():
        path = os.path.join(models_dir, file)
        
        if not os.path.exists(path):
            # Try with .h5 extension for backward compatibility
            h5_path = os.path.join(models_dir, file.replace('.keras', '.h5'))
            if os.path.exists(h5_path):
                print(f"{Fore.YELLOW}Found {name} model with .h5 extension instead of .keras{Style.RESET_ALL}")
                path = h5_path
            else:
                print(f"{Fore.RED}Model {name} not found at {path}{Style.RESET_ALL}")
                continue
        
        print(f"\n{Fore.CYAN}=== Diagnosing {name.upper()} model ==={Style.RESET_ALL}")
        diag = diagnose_model_shape_mismatch(path)
        
        if diag["success"]:
            print(f"Model architecture summary:")
            print(get_model_summary(path)[:500] + "...")  # Print truncated summary
            
            # For LSTM models, check forecast horizon
            if name.startswith('lstm'):
                horizon = diag.get('model_horizon', 'unknown')
                print(f"{Fore.YELLOW}This model has a forecast horizon of: {horizon}{Style.RESET_ALL}")
                if horizon != 24:
                    print(f"{Fore.GREEN}Recommendation: Use forecast_horizon={horizon} in HybridCNNLSTM{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
