import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_time_series_quality(data, column_name='power_output', save_path=None):
    """
    Analyze the quality of a time series and report issues
    
    Args:
        data: DataFrame containing time series data
        column_name: Name of the column to analyze
        save_path: Path to save the diagnostic plot (if None, just displays)
    
    Returns:
        Dictionary with quality metrics and issues found
    """
    if column_name not in data.columns:
        return {'error': f"Column {column_name} not found in data"}
    
    series = data[column_name]
    
    # Calculate quality metrics
    metrics = {
        'length': len(series),
        'nan_count': series.isna().sum(),
        'nan_percent': series.isna().mean() * 100,
        'zero_count': (series == 0).sum(),
        'zero_percent': (series == 0).mean() * 100,
        'negative_count': (series < 0).sum(),
        'negative_percent': (series < 0).mean() * 100,
        'min': series.min(),
        'max': series.max(),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std()
    }
    
    # Check for specific issues
    issues = []
    
    if metrics['nan_percent'] > 5:
        issues.append(f"High percentage of NaN values: {metrics['nan_percent']:.2f}%")
    
    if metrics['zero_percent'] > 50:
        issues.append(f"High percentage of zero values: {metrics['zero_percent']:.2f}%")
    
    if metrics['negative_count'] > 0:
        issues.append(f"Contains negative values: {metrics['negative_count']} ({metrics['negative_percent']:.2f}%)")
    
    # Check for outliers using IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    metrics['outlier_count'] = len(outliers)
    metrics['outlier_percent'] = len(outliers) / len(series) * 100
    
    if metrics['outlier_percent'] > 5:
        issues.append(f"High percentage of outliers: {metrics['outlier_percent']:.2f}%")
    
    # Plot the time series with issues highlighted
    plt.figure(figsize=(12, 8))
    
    # Main time series
    plt.plot(series.values, alpha=0.7, label='Original Data')
    
    # Highlight issues
    if metrics['nan_count'] > 0:
        nan_indices = np.where(series.isna())[0]
        plt.scatter(nan_indices, np.zeros(len(nan_indices)), color='red', alpha=0.5, label='NaN Values')
    
    if metrics['outlier_count'] > 0:
        outlier_indices = series.index[series.isin(outliers)]
        plt.scatter(outlier_indices, series[outlier_indices], color='orange', alpha=0.7, label='Outliers')
    
    # Add bounds
    if not np.isnan(lower_bound) and not np.isnan(upper_bound):
        plt.axhline(y=lower_bound, color='g', linestyle='--', alpha=0.5, label='Lower Bound')
        plt.axhline(y=upper_bound, color='g', linestyle='--', alpha=0.5, label='Upper Bound')
    
    plt.title(f'Time Series Analysis: {column_name}')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics
    info_text = "\n".join([
        f"Length: {metrics['length']}",
        f"NaN: {metrics['nan_percent']:.2f}%",
        f"Zeros: {metrics['zero_percent']:.2f}%",
        f"Outliers: {metrics['outlier_percent']:.2f}%",
        f"Min: {metrics['min']:.2f}",
        f"Max: {metrics['max']:.2f}",
        f"Mean: {metrics['mean']:.2f}"
    ])
    
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    metrics['issues'] = issues
    return metrics

def check_processed_data(X, y, scaler=None):
    """
    Check the quality of processed data arrays for model training
    
    Args:
        X: Input data array
        y: Target data array
        scaler: Optional scaler to check inverse transform
    
    Returns:
        Dictionary with data quality metrics
    """
    def check_array(arr, name):
        return {
            'shape': arr.shape,
            'nan_percent': np.isnan(arr).mean() * 100,
            'inf_percent': np.isinf(arr).mean() * 100,
            'zero_percent': (arr == 0).mean() * 100,
            'min': np.nanmin(arr),
            'max': np.nanmax(arr),
            'mean': np.nanmean(arr),
            'std': np.nanstd(arr)
        }
    
    results = {
        'X': check_array(X, 'X'),
        'y': check_array(y, 'y')
    }
    
    # Check for issues that could cause evaluation problems
    issues = []
    
    if results['y']['nan_percent'] > 0:
        issues.append(f"Target contains {results['y']['nan_percent']:.2f}% NaN values")
    
    if results['y']['inf_percent'] > 0:
        issues.append(f"Target contains {results['y']['inf_percent']:.2f}% infinite values")
    
    if results['y']['zero_percent'] > 90:
        issues.append(f"Target contains {results['y']['zero_percent']:.2f}% zero values")
    
    if scaler:
        try:
            # Test inverse transform of a simple array
            test_data = np.array([[0.5] * scaler.scale_.shape[0]])
            inv_test = scaler.inverse_transform(test_data)
            results['scaler_test'] = {
                'input': test_data.tolist(),
                'output': inv_test.tolist(),
                'working': True
            }
        except Exception as e:
            results['scaler_test'] = {
                'error': str(e),
                'working': False
            }
            issues.append(f"Scaler inverse_transform test failed: {str(e)}")
    
    results['issues'] = issues
    return results

def inspect_model_predictions(y_true, y_pred, title='Model Predictions vs Actual', save_path=None):
    """
    Create diagnostic plots to inspect model predictions against actual values
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        title: Title for the plot
        save_path: Path to save the plot (if None, just displays)
    """
    # Flatten arrays if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Create comparison plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axs[0, 0].plot(y_true, label='Actual')
    axs[0, 0].plot(y_pred, label='Predicted')
    axs[0, 0].set_title('Time Series Comparison')
    axs[0, 0].set_xlabel('Time Index')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axs[0, 1].scatter(y_true, y_pred, alpha=0.5)
    axs[0, 1].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    axs[0, 1].set_title('Prediction vs Actual Scatter Plot')
    axs[0, 1].set_xlabel('Actual')
    axs[0, 1].set_ylabel('Predicted')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Error histogram
    error = y_pred - y_true
    axs[1, 0].hist(error, bins=30, alpha=0.7)
    axs[1, 0].axvline(x=0, color='r', linestyle='--')
    axs[1, 0].set_title('Error Distribution')
    axs[1, 0].set_xlabel('Prediction Error')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Error vs actual
    axs[1, 1].scatter(y_true, error, alpha=0.5)
    axs[1, 1].axhline(y=0, color='r', linestyle='--')
    axs[1, 1].set_title('Error vs Actual Value')
    axs[1, 1].set_xlabel('Actual Value')
    axs[1, 1].set_ylabel('Error')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Calculate and display metrics
    metrics = {
        'RMSE': np.sqrt(np.nanmean((y_true - y_pred)**2)),
        'MAE': np.nanmean(np.abs(y_true - y_pred)),
        'Mean Error': np.nanmean(error),
        'Error Std': np.nanstd(error)
    }
    
    # Add correlation if not all values are the same
    if np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        # Use Pearson correlation only for finite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(mask) > 1:  # Need at least 2 points for correlation
            metrics['Correlation'] = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
        else:
            metrics['Correlation'] = float('nan')
    else:
        metrics['Correlation'] = float('nan')
    
    # Add text with metrics
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return metrics
