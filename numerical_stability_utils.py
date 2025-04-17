import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def check_numerical_stability(model, X_sample):
    """
    Analyze model inputs/outputs to check for numerical stability issues
    
    Args:
        model: Keras model to check
        X_sample: Sample input data
        
    Returns:
        Dict with numerical stability information
    """
    results = {}
    
    # Get model's internal layers
    layers = model.layers
    
    # Check input data
    results["input"] = {
        "shape": X_sample.shape,
        "min": np.min(X_sample),
        "max": np.max(X_sample),
        "mean": np.mean(X_sample),
        "std": np.std(X_sample),
        "has_nan": np.isnan(X_sample).any(),
        "has_inf": np.isinf(X_sample).any(),
        "zeros_percent": (X_sample == 0).mean() * 100
    }
    
    # Check weights and activations
    intermediate_models = []
    layer_outputs = []
    
    # Create intermediate models to extract layer outputs
    for i in range(1, len(layers)):
        intermediate_model = tf.keras.Model(inputs=model.input, 
                                           outputs=layers[i].output)
        intermediate_models.append(intermediate_model)
    
    # Compute outputs for each layer
    layer_stats = []
    try:
        for i, im in enumerate(intermediate_models):
            layer_name = layers[i+1].name
            
            # Skip certain layers that don't transform data
            if "reshape" in layer_name or "input" in layer_name:
                continue
                
            output = im.predict(X_sample)
            layer_outputs.append(output)
            
            # Calculate stats
            layer_stat = {
                "name": layer_name,
                "shape": output.shape,
                "min": np.min(output),
                "max": np.max(output),
                "mean": np.mean(output),
                "std": np.std(output),
                "has_nan": np.isnan(output).any(),
                "has_inf": np.isinf(output).any(),
                "zeros_percent": (output == 0).mean() * 100,
                "dist": "normal" if 0.1 < abs(np.mean(output)) < 10 and 0.1 < np.std(output) < 10 else "abnormal"
            }
            layer_stats.append(layer_stat)
    except Exception as e:
        print(f"Error during layer inspection: {e}")
    
    results["layers"] = layer_stats
    
    # Check weights for NaN/Inf values
    weight_issues = []
    for layer in layers:
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            if np.isnan(w).any() or np.isinf(w).any():
                weight_issues.append({
                    "layer": layer.name,
                    "weight_idx": i,
                    "has_nan": np.isnan(w).any(),
                    "has_inf": np.isinf(w).any()
                })
    
    results["weight_issues"] = weight_issues
    
    # Recommend improvements
    suggestions = []
    
    if results["input"]["has_nan"] or results["input"]["has_inf"]:
        suggestions.append("Input data contains NaN or Inf values. Clean input data before training.")
    
    for ls in layer_stats:
        if ls["has_nan"] or ls["has_inf"]:
            suggestions.append(f"Layer {ls['name']} produces NaN or Inf values. Consider adding BatchNormalization or reducing learning rate.")
        
        if ls["max"] > 100 or ls["min"] < -100:
            suggestions.append(f"Layer {ls['name']} has extreme values. Consider adding BatchNormalization or weight regularization.")
    
    if weight_issues:
        suggestions.append("Some model weights contain NaN or Inf values. Model is unstable - restart training with a smaller learning rate.")
    
    results["suggestions"] = suggestions
    
    return results

def plot_numerical_stability(model, X_sample, save_path=None):
    """
    Plot numerical characteristics of the model layers
    
    Args:
        model: Keras model to analyze
        X_sample: Sample input data
        save_path: Optional path to save the plot
    """
    layers = model.layers
    intermediate_models = []
    
    # Create intermediate models
    for i in range(1, len(layers)):
        if not any(skip in layers[i].name for skip in ['reshape', 'input', 'lambda']):
            try:
                intermediate_model = tf.keras.Model(inputs=model.input, 
                                                  outputs=layers[i].output)
                intermediate_models.append((layers[i].name, intermediate_model))
            except:
                continue
    
    # Create plots
    if not intermediate_models:
        print("No suitable layers found for plotting")
        return
        
    fig, axs = plt.subplots(len(intermediate_models), 3, figsize=(18, 4*len(intermediate_models)))
    
    # Handle single layer case
    if len(intermediate_models) == 1:
        axs = np.array([axs])
    
    for i, (name, im) in enumerate(intermediate_models):
        try:
            # Get layer output
            output = im.predict(X_sample)
            
            # If output is multi-dimensional, flatten for histogram
            if len(output.shape) > 2:
                flat_output = output.flatten()
            else:
                flat_output = output.reshape(-1)
            
            # Skip if all values are the same
            if np.all(flat_output == flat_output[0]):
                axs[i, 0].text(0.5, 0.5, "All values are identical", 
                               ha='center', va='center')
                axs[i, 1].text(0.5, 0.5, "All values are identical", 
                               ha='center', va='center')
                axs[i, 2].text(0.5, 0.5, "All values are identical", 
                               ha='center', va='center')
                continue
                
            # Histogram of layer outputs
            axs[i, 0].hist(flat_output, bins=50)
            axs[i, 0].set_title(f"{name} - Output Distribution")
            
            # Check if we have enough samples to plot a meaningful Q-Q plot
            if len(flat_output) > 10:
                # Q-Q plot (to check for normality)
                from scipy import stats
                stats.probplot(flat_output, plot=axs[i, 1])
                axs[i, 1].set_title(f"{name} - Q-Q Plot")
            else:
                axs[i, 1].text(0.5, 0.5, "Not enough data points for Q-Q plot", 
                              ha='center', va='center')
            
            # Show layer stats
            stats_text = "\n".join([
                f"Min: {np.min(flat_output):.4f}",
                f"Max: {np.max(flat_output):.4f}", 
                f"Mean: {np.mean(flat_output):.4f}", 
                f"Std: {np.std(flat_output):.4f}",
                f"NaNs: {'Yes' if np.isnan(flat_output).any() else 'No'}",
                f"Infs: {'Yes' if np.isinf(flat_output).any() else 'No'}",
                f"Zeros: {(flat_output == 0).mean():.2%}"
            ])
            axs[i, 2].text(0.1, 0.5, stats_text, va='center')
            axs[i, 2].axis('off')
            axs[i, 2].set_title(f"{name} - Statistics")
            
        except Exception as e:
            print(f"Error plotting layer {name}: {e}")
            for j in range(3):
                axs[i, j].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def diagnose_nan_loss(model, X, y):
    """
    Diagnose why a model might be producing NaN loss
    
    Args:
        model: Keras model to analyze
        X: Input data
        y: Target data
        
    Returns:
        Dict with diagnosis information
    """
    # Check input and output data
    input_issues = []
    output_issues = []
    
    # Check for NaN/Inf in inputs
    if np.isnan(X).any():
        input_issues.append("Inputs contain NaN values")
    if np.isinf(X).any():
        input_issues.append("Inputs contain Inf values")
        
    # Check for extreme values in inputs
    x_min, x_max = np.min(X), np.max(X)
    if x_max > 1e10 or x_min < -1e10:
        input_issues.append(f"Input contains extreme values: min={x_min}, max={x_max}")
    
    # Check for NaN/Inf in outputs
    if np.isnan(y).any():
        output_issues.append("Targets contain NaN values")
    if np.isinf(y).any():
        output_issues.append("Targets contain Inf values")
        
    # Check model weights for NaN/Inf
    model_issues = []
    for layer in model.layers:
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            if np.isnan(w).any():
                model_issues.append(f"Layer {layer.name} weight {i} contains NaN values")
            if np.isinf(w).any():
                model_issues.append(f"Layer {layer.name} weight {i} contains Inf values")
    
    # Check if model is using learning rate that's too high
    optimizer_config = model.optimizer.get_config()
    if 'learning_rate' in optimizer_config and optimizer_config['learning_rate'] > 0.01:
        model_issues.append(f"Learning rate may be too high: {optimizer_config['learning_rate']}")
    
    # Check for common issues with CNN architecture
    architecture_issues = []
    
    # Check for absence of BatchNormalization
    has_batch_norm = any(['batch_normalization' in layer.name for layer in model.layers])
    if not has_batch_norm:
        architecture_issues.append("Model doesn't use BatchNormalization - consider adding it for training stability")
    
    # Check for absence of gradient clipping
    if hasattr(model.optimizer, 'clipnorm') and not model.optimizer.clipnorm:
        architecture_issues.append("Gradient clipping is not enabled - consider adding clipnorm to optimizer")
    
    # Recommendations
    recommendations = []
    
    if input_issues:
        recommendations.append("Clean input data to remove NaN, Inf values and normalize to a reasonable range")
    
    if output_issues:
        recommendations.append("Clean target data to remove NaN, Inf values")
    
    if model_issues:
        recommendations.append("Reset model weights and train with a lower learning rate (try 1e-4)")
    
    if architecture_issues:
        for issue in architecture_issues:
            recommendations.append(issue)
    
    # If no specific issues found, provide general recommendations
    if not (input_issues or output_issues or model_issues or architecture_issues):
        recommendations.extend([
            "Try reducing batch size",
            "Enable gradient clipping in optimizer (clipnorm=1.0)",
            "Add a TerminateOnNaN callback and EarlyStopping",
            "Try a different weight initialization",
            "Use a more stable loss function"
        ])
    
    return {
        "input_issues": input_issues,
        "output_issues": output_issues, 
        "model_issues": model_issues,
        "architecture_issues": architecture_issues,
        "recommendations": recommendations
    }

# Usage example
if __name__ == "__main__":
    # Load the model and sample data
    model = tf.keras.models.load_model('models/hybrid_model_cnn.keras')
    
    # Create some sample data matching the input shape
    input_shape = model.input_shape[1:]
    X_sample = np.random.rand(10, *input_shape)
    
    # Check numerical stability
    results = check_numerical_stability(model, X_sample)
    
    # Print suggestions
    for suggestion in results["suggestions"]:
        print(f"• {suggestion}")
        
    # Plot stability analysis
    plot_numerical_stability(model, X_sample, save_path='plots/numerical_stability.png')
