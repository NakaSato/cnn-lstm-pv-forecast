import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_classifier(input_shape):
    """
    Build CNN model for weather classification (sunny/cloudy)
    """
    # Use the Input layer first to avoid the warning
    inputs = Input(shape=input_shape)
    
    # Add batch normalization to stabilize inputs
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # First conv block with reduced filter count
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Second conv block
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a smaller learning rate and add gradient clipping
    # Fix: Use only clipnorm, not both clipnorm and clipvalue
    optimizer = Adam(
        learning_rate=0.0001,  # Reduced learning rate
        clipnorm=1.0          # Add gradient clipping (removed clipvalue)
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def build_lstm_sunny(input_shape, output_shape):
    """
    Build LSTM model for sunny day power prediction
    
    Args:
        input_shape: tuple (sequence_length, feature_dims)
        output_shape: tuple (forecast_horizon, output_feature_dims)
    """
    # Create a model using functional API to avoid warnings
    inputs = Input(shape=input_shape)
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    x = Dense(output_shape[0] * output_shape[1])(x)
    outputs = tf.keras.layers.Reshape(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001)
    )
    
    return model

def build_lstm_cloudy(input_shape, output_shape):
    """
    Build LSTM model for cloudy day power prediction
    
    Args:
        input_shape: tuple (sequence_length, feature_dims)
        output_shape: tuple (forecast_horizon, output_feature_dims)
    """
    # Create a model using functional API to avoid warnings
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(output_shape[0] * output_shape[1])(x)
    outputs = tf.keras.layers.Reshape(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001)
    )
    
    return model

class HybridCNNLSTM:
    """
    Hybrid model that combines CNN for classification and LSTMs for prediction
    """
    def __init__(self, sequence_length, forecast_horizon, feature_dims=None):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon if forecast_horizon is not None else 7  # Default to 7 if None
        self.feature_dims = feature_dims if feature_dims is not None else 1
        
        self.output_feature_dims = self.feature_dims
        
        # Only initialize output_shape if forecast_horizon is defined
        if forecast_horizon is not None:
            self.output_shape = (forecast_horizon, self.output_feature_dims)
            
            self.cnn = build_cnn_classifier(input_shape=(sequence_length, self.feature_dims))
            self.lstm_sunny = build_lstm_sunny(
                input_shape=(sequence_length, self.feature_dims),
                output_shape=self.output_shape
            )
            self.lstm_cloudy = build_lstm_cloudy(
                input_shape=(sequence_length, self.feature_dims),
                output_shape=self.output_shape
            )
        else:
            # If forecast_horizon is None, don't build models yet - they'll be loaded from files
            self.cnn = None
            self.lstm_sunny = None
            self.lstm_cloudy = None
            
    def train_cnn(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        print(f"CNN input shape: {X_train.shape}")
        print(f"CNN target shape: {y_train.shape}")
        
        # Check for NaN or infinite values in training data
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("Warning: Training data contains NaN or inf values. Cleaning data...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Add a callback for early stopping to prevent NaN loss
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        
        # Add NaN callback to stop training if NaN is detected
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        
        return self.cnn.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=[early_stopping, nan_callback]
        )
    
    def train_lstm_sunny(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        print(f"LSTM sunny input shape: {X_train.shape}")
        print(f"LSTM sunny target shape: {y_train.shape}")
        
        if y_train.shape[-1] != self.output_feature_dims:
            print(f"Updating output feature dimensions from {self.output_feature_dims} to {y_train.shape[-1]}")
            self.output_feature_dims = y_train.shape[-1]
            self.output_shape = (self.forecast_horizon, self.output_feature_dims)
            
            self.lstm_sunny = build_lstm_sunny(
                input_shape=(self.sequence_length, self.feature_dims),
                output_shape=self.output_shape
            )
        
        return self.lstm_sunny.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
    
    def train_lstm_cloudy(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        print(f"LSTM cloudy input shape: {X_train.shape}")
        print(f"LSTM cloudy target shape: {y_train.shape}")
        
        if y_train.shape[-1] != self.output_feature_dims:
            print(f"Updating output feature dimensions from {self.output_feature_dims} to {y_train.shape[-1]}")
            self.output_feature_dims = y_train.shape[-1]
            self.output_shape = (self.forecast_horizon, self.output_feature_dims)
            
            self.lstm_cloudy = build_lstm_cloudy(
                input_shape=(self.sequence_length, self.feature_dims),
                output_shape=self.output_shape
            )
        
        return self.lstm_cloudy.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
    
    def predict(self, X):
        if len(X.shape) == 3 and X.shape[-1] != 1:
            X_cnn = X  # No need to expand dimensions if we already have features
        else:
            X_cnn = X
        
        # Check sequence length and adjust if needed
        expected_seq_len = self.cnn.input_shape[1]
        actual_seq_len = X_cnn.shape[1]
        
        if expected_seq_len != actual_seq_len:
            print(f"WARNING: Input sequence length mismatch. Expected: {expected_seq_len}, Got: {actual_seq_len}")
            if expected_seq_len > actual_seq_len:
                # Pad with zeros if the input is too short
                padding_length = expected_seq_len - actual_seq_len
                padding_shape = (X_cnn.shape[0], padding_length, X_cnn.shape[2])
                padding = np.zeros(padding_shape)
                X_cnn = np.concatenate([X_cnn, padding], axis=1)
                print(f"Padded input sequence from length {actual_seq_len} to {X_cnn.shape[1]}")
            else:
                # Truncate if the input is too long
                X_cnn = X_cnn[:, :expected_seq_len, :]
                print(f"Truncated input sequence from length {actual_seq_len} to {X_cnn.shape[1]}")
        
        # Check for NaN values in input and replace them
        if np.isnan(X_cnn).any():
            print("Warning: Input data contains NaN values. Replacing with zeros for prediction.")
            X_cnn = np.nan_to_num(X_cnn, nan=0.0, posinf=1.0, neginf=0.0)
            
        # Get weather predictions
        try:
            # Add additional check to ensure input is finite
            if not np.all(np.isfinite(X_cnn)):
                print("Warning: Input contains non-finite values. Fixing before prediction.")
                X_cnn = np.clip(X_cnn, -10, 10)  # Clip to reasonable range
                
            weather_pred = self.cnn.predict(X_cnn)
        except Exception as e:
            print(f"Error during CNN prediction: {e}")
            print(f"Input shape: {X_cnn.shape}, Expected input shape for CNN: {self.cnn.input_shape}")
            # As a fallback, create random weather predictions
            weather_pred = np.random.rand(len(X_cnn), 2)
            
        predictions = np.zeros((len(X), self.forecast_horizon, self.output_feature_dims))
        
        # Create a clean copy of X for LSTM input
        X_clean = X.copy()
        if np.isnan(X_clean).any():
            X_clean = np.nan_to_num(X_clean)
        
        # Check and adjust sequence length for LSTM models as well
        lstm_expected_seq_len = self.lstm_sunny.input_shape[1]
        if lstm_expected_seq_len != actual_seq_len:
            if lstm_expected_seq_len > actual_seq_len:
                # Pad with zeros
                padding_length = lstm_expected_seq_len - actual_seq_len
                padding_shape = (X_clean.shape[0], padding_length, X_clean.shape[2])
                padding = np.zeros(padding_shape)
                X_clean = np.concatenate([X_clean, padding], axis=1)
                print(f"Padded LSTM input sequence to length {X_clean.shape[1]}")
            else:
                # Truncate
                X_clean = X_clean[:, :lstm_expected_seq_len, :]
                print(f"Truncated LSTM input sequence to length {X_clean.shape[1]}")
        
        for i, sample in enumerate(X_clean):
            try:
                sample_reshaped = sample.reshape(1, self.sequence_length, self.feature_dims)
                
                if np.argmax(weather_pred[i]) == 1:
                    pred = self.lstm_sunny.predict(sample_reshaped)
                else:
                    pred = self.lstm_cloudy.predict(sample_reshaped)
                    
                if np.isnan(pred).any():
                    print(f"Warning: LSTM produced NaN predictions for sample {i}. Using zeros.")
                    pred = np.zeros_like(pred)
                    
                predictions[i] = pred[0]
            except Exception as e:
                print(f"Error predicting sample {i}: {e}")
                # Use zeros as fallback
                predictions[i] = np.zeros((self.forecast_horizon, self.output_feature_dims))
        
        return predictions
    
    def save_models(self, path_prefix):
        """Save the models using the modern .keras format"""
        self.cnn.save(f"{path_prefix}_cnn.keras")
        self.lstm_sunny.save(f"{path_prefix}_lstm_sunny.keras")
        self.lstm_cloudy.save(f"{path_prefix}_lstm_cloudy.keras")
    
    def load_models(self, path_prefix):
        """Load models from the modern .keras format"""
        try:
            # First try to load with .keras extension
            self.cnn = tf.keras.models.load_model(f"{path_prefix}_cnn.keras")
            self.lstm_sunny = tf.keras.models.load_model(f"{path_prefix}_lstm_sunny.keras")
            self.lstm_cloudy = tf.keras.models.load_model(f"{path_prefix}_lstm_cloudy.keras")
            
            # Update forecast_horizon based on loaded model
            # Get the output shape from the sunny LSTM model
            if self.lstm_sunny is not None and hasattr(self.lstm_sunny, 'output_shape'):
                model_forecast_horizon = self.lstm_sunny.output_shape[1]
                if self.forecast_horizon != model_forecast_horizon:
                    print(f"Updating forecast horizon from {self.forecast_horizon} to {model_forecast_horizon} based on loaded model")
                    self.forecast_horizon = model_forecast_horizon
                    self.output_shape = (model_forecast_horizon, self.output_feature_dims)
                    
        except (OSError, IOError):
            # Fallback to the legacy .h5 format for backward compatibility
            print("Could not find models in .keras format. Trying legacy .h5 format...")
            self.cnn = tf.keras.models.load_model(f"{path_prefix}_cnn.h5")
            self.lstm_sunny = tf.keras.models.load_model(f"{path_prefix}_lstm_sunny.h5")
            self.lstm_cloudy = tf.keras.models.load_model(f"{path_prefix}_lstm_cloudy.h5")
            
            # Update forecast_horizon based on loaded model
            if self.lstm_sunny is not None and hasattr(self.lstm_sunny, 'output_shape'):
                model_forecast_horizon = self.lstm_sunny.output_shape[1]
                if self.forecast_horizon != model_forecast_horizon:
                    print(f"Updating forecast horizon from {self.forecast_horizon} to {model_forecast_horizon} based on loaded model")
                    self.forecast_horizon = model_forecast_horizon
                    self.output_shape = (model_forecast_horizon, self.output_feature_dims)
