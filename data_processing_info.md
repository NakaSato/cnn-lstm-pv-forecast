# PV Inverter Data Processing Guide

## Data Format

The CNN-LSTM PV forecasting system works with inverter data in CSV format. The standard format contains the following columns:

| Column       | Description            | Data Type | Units               |
| ------------ | ---------------------- | --------- | ------------------- |
| Timestamp    | Combined date and time | Datetime  | YYYY-MM-DD HH:MM:SS |
| Date         | Calendar date          | Date      | YYYY-MM-DD          |
| Time         | Time of day            | Time      | HH:MM:SS            |
| AC_Power     | AC power output        | Float     | kW                  |
| DC_Power     | DC power input         | Float     | kW                  |
| Voltage_AC   | AC output voltage      | Float     | Volts               |
| Current_AC   | AC output current      | Float     | Amperes             |
| Frequency    | Grid frequency         | Float     | Hz                  |
| Efficiency   | Inverter efficiency    | Float     | Ratio (0-1)         |
| Status_Flags | Status/error codes     | Integer   | -                   |

## Data Preprocessing

During preprocessing, the system:

1. Maps these columns to standardized internal names:

   - `AC_Power` → `power_output`
   - `DC_Power` → `dc_power`
   - And so on

2. Performs several transformations:

   - Removes outliers using IQR method
   - Interpolates missing values
   - Normalizes data to 0-1 range
   - Extracts time features (hour of day, day of year, etc.)
   - Creates rolling statistics (mean, standard deviation, etc.)

3. Detects weather patterns:
   - Uses power output patterns to detect sunny vs. cloudy conditions
   - This enables specialized prediction models for different weather conditions

## Adding New Data

To add new inverter data:

1. Ensure your CSV contains at least the `Timestamp` and `AC_Power` columns
2. Additional columns improve prediction accuracy
3. Place the file in the `data/` directory
4. Use the standard naming format: `INVERTER_XX.csv` where XX is the inverter number

## Handling Different Data Formats

If your data has different column names, update the `column_mapping` dictionary in `data_preprocessing.py` to map your column names to the standard names used by the system.
