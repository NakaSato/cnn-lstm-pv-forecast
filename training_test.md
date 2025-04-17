### Main Configuration Parameters in train.py:

1. **Random Seed**

   - `RANDOM_SEED = 42` - Controls the reproducibility of the training process

2. **Sequence Parameters**

   - `SEQUENCE_LENGTH = 24` - Number of time steps in the input sequence (currently set for one day of hourly data)
   - `FORECAST_HORIZON = 7` - Number of time steps to forecast (predicts the next day)

3. **Training Parameters**
   - `EPOCHS_CNN = 50` - Number of training epochs for the CNN model
   - `EPOCHS_LSTM = 100` - Number of training epochs for the LSTM model
   - `BATCH_SIZE = 32` - Batch size used during training
   - `VALIDATION_SPLIT = 0.2` - Fraction of data used for validation

### Additional Parameters That Can Be Modified:

1. **Model Architecture Parameters** (in models.py)

   - CNN layers, filters, and kernel sizes
   - LSTM units and layers
   - Dropout rates
   - Activation functions

2. **Optimizer Parameters**

   - Learning rate
   - Optimizer type (currently using default)
   - Loss functions

3. **Data Processing Parameters** (in data_preprocessing.py)
   - Data normalization method
   - Feature selection
   - Handling of missing values

### How to Adjust These Parameters:

To adjust these parameters, you can simply edit the constants at the beginning of the train.py file. For more extensive architecture changes, you would need to modify the model definitions in models.py.

You could also make the training script more flexible by adding command-line arguments for these parameters, allowing you to experiment with different values without changing the code each time.

Would you like me to explain any specific parameter in more detail or help you implement a more flexible training process with command-line arguments?
