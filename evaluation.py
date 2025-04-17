import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE (Mean Absolute Percentage Error) with proper handling of edge cases"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Check if arrays are empty
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Empty arrays passed to MAPE calculation")
        return float('nan')
    
    # Filter out pairs where true value is zero to avoid division by zero
    mask = np.abs(y_true) > 1e-10  # Small threshold instead of exact zero
    
    # If all true values are zero or near-zero, MAPE is undefined
    if np.sum(mask) == 0:
        print("Warning: All true values are zero or near-zero, MAPE is undefined")
        # Return a large value instead of NaN to indicate poor performance
        return 999.99
    
    # Calculate MAPE only on non-zero true values
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Check for overflow or invalid result
    if not np.isfinite(mape):
        print("Warning: MAPE calculation resulted in non-finite value")
        return 999.99
        
    return mape

def evaluate_model(y_true, y_pred):
    """Evaluate model using multiple metrics with robust error handling"""
    # Convert to flat arrays if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Print some basic statistics
    print(f"Data points: {len(y_true)}")
    print(f"y_true stats: min={np.nanmin(y_true):.4f}, max={np.nanmax(y_true):.4f}, mean={np.nanmean(y_true):.4f}")
    print(f"y_pred stats: min={np.nanmin(y_pred):.4f}, max={np.nanmax(y_pred):.4f}, mean={np.nanmean(y_pred):.4f}")
    
    # Handle NaN values by removing them
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # If we have NaN values, filter them out and warn the user
    if not np.all(mask):
        nan_count = len(y_true) - np.sum(mask)
        print(f"Warning: Found {nan_count} NaN values. These will be removed for evaluation.")
        
        # Only filter if we don't lose all data
        if np.sum(mask) > 0:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        else:
            # As a last resort, replace NaNs with zeros instead of filtering
            print("Warning: All values would be NaN. Replacing NaNs with zeros for evaluation.")
            y_true = np.nan_to_num(y_true)
            y_pred = np.nan_to_num(y_pred)
    
    # Filter out any infinity values
    mask_finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask_finite):
        inf_count = len(y_true) - np.sum(mask_finite)
        print(f"Warning: Found {inf_count} infinite values. These will be removed for evaluation.")
        if np.sum(mask_finite) > 0:
            y_true = y_true[mask_finite]
            y_pred = y_pred[mask_finite]
        else:
            print("Warning: All values would be infinite. Using zeros for evaluation.")
            y_true = np.zeros_like(y_true)
            y_pred = np.zeros_like(y_pred)
    
    # If we still have data after handling NaNs, calculate metrics
    if len(y_true) > 0:
        try:
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'MAPE': mape,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'MAPE': float('nan'),
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'R2': float('nan')
            }
    else:
        print("Error: No valid data points for evaluation after filtering NaNs.")
        # Return placeholder metrics
        metrics = {
            'MAPE': float('nan'),
            'RMSE': float('nan'),
            'MAE': float('nan'),
            'R2': float('nan')
        }
    
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"R²: {metrics['R2']:.4f}")
