import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
df = pd.read_csv('./processed_data/solar_data_20250416_011748.csv',
                 parse_dates=['Time'],
                 date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
                 low_memory=False)
# Check for missing values

# Data Preparation
# Select relevant columns
inverter_data = df[['Time', 'Pac(W)', 'Ppv(W)', 'VacR(V)', 'VacS(V)', 'VacT(V)', 
                   'VacRS(V)', 'VacST(V)', 'VacTR(V)', 'IacR(A)', 'IacS(A)', 
                   'IacT(A)', 'Fac(Hz)', 'INVTemp(℃)', 'PIDStatus']].copy()

# Rename columns using the mapping provided
inverter_data = inverter_data.rename(columns={
    'Time': 'Timestamp',
    'Pac(W)': 'AC_Power',
    'Ppv(W)': 'DC_Power', 
    'VacR(V)': 'Voltage_AC_R',
    'VacS(V)': 'Voltage_AC_S',
    'VacT(V)': 'Voltage_AC_T',
    'VacRS(V)': 'Voltage_AC_RS',
    'VacST(V)': 'Voltage_AC_ST',
    'VacTR(V)': 'Voltage_AC_TR',
    'IacR(A)': 'Current_AC_R',
    'IacS(A)': 'Current_AC_S',
    'IacT(A)': 'Current_AC_T',
    'Fac(Hz)': 'Frequency',
    'INVTemp(℃)': 'Temperature',
    'PIDStatus': 'Status'
})

# Calculate average voltage and current for simplicity
inverter_data['Voltage_AC'] = inverter_data[['Voltage_AC_R', 'Voltage_AC_S', 'Voltage_AC_T']].mean(axis=1)
inverter_data['Current_AC'] = inverter_data[['Current_AC_R', 'Current_AC_S', 'Current_AC_T']].mean(axis=1)
inverter_data.columns = ['Timestamp', 'AC_Power', 'DC_Power', 'Voltage_AC', 'Current_AC', 'Frequency', 'Temperature', 'Status']

# Convert timestamp to datetime and set as index
inverter_data['Timestamp'] = pd.to_datetime(inverter_data['Timestamp'])
inverter_data.set_index('Timestamp', inplace=True)

# Filter out nighttime data (when AC_Power is 0)
inverter_data = inverter_data[inverter_data['AC_Power'] > 0]

# Feature Engineering
# Calculate efficiency (DC to AC conversion)
inverter_data['Efficiency'] = inverter_data['AC_Power'] / inverter_data['DC_Power'] * 100

# Handle any infinite values from division
inverter_data.replace([np.inf, -np.inf], np.nan, inplace=True)
inverter_data.dropna(inplace=True)

# Visualize the data
plt.figure(figsize=(15, 8))
plt.plot(inverter_data.index, inverter_data['AC_Power'], label='AC Power (W)')
plt.plot(inverter_data.index, inverter_data['DC_Power'], label='DC Power (W)')
plt.title('Solar Inverter Power Output')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.legend()
plt.grid()
plt.show()

# Prepare data for LSTM
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), :])
        y.append(data[i+look_back, 0])  # Predicting AC_Power
    return np.array(X), np.array(y)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(inverter_data[['AC_Power', 'DC_Power', 'Voltage_AC', 
                                                'Current_AC', 'Frequency', 'Temperature', 'Efficiency']])

# Split into train and test sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Create sequences
look_back = 60  # Number of previous time steps to use
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stop],
                    verbose=1)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(np.concatenate((train_predict, 
                                                       np.zeros((train_predict.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
y_train = scaler.inverse_transform(np.concatenate((y_train.reshape(-1,1), 
                                                np.zeros((y_train.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

test_predict = scaler.inverse_transform(np.concatenate((test_predict, 
                                                      np.zeros((test_predict.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1,1), 
                                               np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

# Calculate RMSE
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(inverter_data.index[look_back:train_size-1], y_train, label='Actual Train')
plt.plot(inverter_data.index[look_back:train_size-1], train_predict, label='Predicted Train')
plt.plot(inverter_data.index[train_size+look_back:-1], y_test, label='Actual Test')
plt.plot(inverter_data.index[train_size+look_back:-1], test_predict, label='Predicted Test')
plt.title('Solar Inverter AC Power Prediction')
plt.xlabel('Time')
plt.ylabel('AC Power (W)')
plt.legend()
plt.grid()
plt.show()

# Save the model for future use
model.save('solar_inverter_lstm.keras')