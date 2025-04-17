# Hybrid CNN-LSTM PV Power Forecasting

This repository implements a hybrid CNN-LSTM model for photovoltaic (PV) power forecasting as described in the research paper. The model combines a CNN for weather classification (sunny vs cloudy) with specialized LSTM models for each weather condition to improve prediction accuracy.

## Key Features

- CNN classifier for weather condition detection
- Separate LSTM models for sunny and cloudy day power generation forecasting
- Data preprocessing with outlier removal and normalization
- Integration with external weather data
- Evaluation metrics: MAPE, RMSE, MAE, R²

## Weather Data Integration

The system now supports integration with external weather data. The expected format is:

| Column                 | Description                          |
| ---------------------- | ------------------------------------ |
| Time                   | Timestamp in format MM/DD/YYYY HH:MM |
| Irradiance (W/m2)      | Solar irradiance                     |
| Env Temp (Celsius)     | Ambient temperature                  |
| Panel Temp (Celsius)   | PV panel temperature                 |
| Wind Speed (m/s)       | Wind speed                           |
| Env Humidity (Celsius) | Environmental humidity               |
| hour                   | Hour of day                          |
| day_of_week            | Day of week (0-6)                    |
| month                  | Month (1-12)                         |
| is_weekend             | Weekend flag (TRUE/FALSE)            |

Additional meteorological and derived features like Heat_Index, Wind_X, Wind_Y will be used if available.

## Project Structure

```
cnn-lstm-pv-forecast/
├── data/               # Data directory for PV time series
│   ├── INVERTER_01.csv  # PV inverter data
│   └── weather-2025-04-04_cleaned.csv  # Weather data
├── models/             # Saved model files
├── plots/              # Generated plots
├── results/            # Prediction results
├── data_preprocessing.py
├── models.py
├── evaluation.py
├── train.py
├── predict.py
├── weather_utils.py    # Weather data processing utilities
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Format

The system uses PV inverter data with the following structure:

| Column       | Description                                  |
| ------------ | -------------------------------------------- |
| Timestamp    | Combined date and time (YYYY-MM-DD HH:MM:SS) |
| Date         | Date in YYYY-MM-DD format                    |
| Time         | Time in HH:MM:SS format                      |
| AC_Power     | AC power output of the inverter (kW)         |
| DC_Power     | DC power input to the inverter (kW)          |
| Voltage_AC   | AC output voltage (V)                        |
| Current_AC   | AC output current (A)                        |
| Frequency    | Grid frequency (Hz)                          |
| Efficiency   | Inverter efficiency ratio                    |
| Status_Flags | Status/error codes from the inverter         |

The preprocessing pipeline automatically handles this format and maps these columns to standardized names for model training.

### Training

```bash
python train.py
```

This will:

1. Load and preprocess the PV data
2. Train the CNN weather classifier
3. Train separate LSTM models for sunny and cloudy days
4. Evaluate the model and save results

### Prediction

```bash
python predict.py
```

This will load the trained models and make predictions on new data.

## Model Performance

Based on the research paper:

- Sunny Days: MAPE = 4.58%, RMSE = 43.87, MAE = 34.00, R² = 0.99
- Cloudy Days: MAPE = 7.06%, RMSE = 9.09, MAE = 6.97, R² = 0.99

## References

This implementation is based on the research methodology described in the paper about the hybrid CNN-LSTM approach for PV power forecasting using only historical PV data.
