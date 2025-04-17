import pandas as pd
import numpy as np
from datetime import datetime
import os
from data.test.dataset.dataset_acquisition import import_solar_data, analyze_solar_data

def save_solar_data(df, output_path=None, format='csv'):
    """
    Save solar data to a file in the specified format
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to save
    output_path : str, optional
        Path where the file should be saved
    format : str, optional
        File format to save ('csv', 'excel', 'pickle', 'parquet')
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # If no output path specified, create one based on current time
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./processed_solar_data_{timestamp}"
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add appropriate file extension if not present
    format = format.lower()
    if not output_path.endswith(f'.{format}'):
        if format == 'excel':
            output_path = f"{output_path}.xlsx"
        else:
            output_path = f"{output_path}.{format}"
    
    # Save in the specified format
    if format == 'csv':
        df.to_csv(output_path, index=False)
        print(f"Data saved as CSV: {output_path}")
    elif format == 'excel':
        df.to_excel(output_path, index=False)
        print(f"Data saved as Excel: {output_path}")
    elif format == 'pickle':
        df.to_pickle(output_path)
        print(f"Data saved as Pickle: {output_path}")
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
        print(f"Data saved as Parquet: {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', 'pickle', or 'parquet'.")
    
    return output_path

def standardize_solar_data(df):
    """
    Standardize solar data fields to ensure consistent units and field names
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw solar data DataFrame
        
    Returns:
    --------
    df : pandas DataFrame
        Standardized DataFrame with consistent units and field names
    """
    # Create a copy to avoid modifying original
    std_df = df.copy()
    
    # Standardized fields with units:
    # Timestamp (UTC/local) - Critical for sequences
    # AC_Power (kW) - Target variable
    # DC_Power (kW) - Useful for loss analysis
    # Voltage_AC (V) - Optional for stability
    # Current_AC (A) - Optional
    # Frequency (Hz) - Optional
    # Efficiency (%) - Useful for degradation
    # Status_Flags - Clean outliers
    
    # Convert power from watts to kilowatts if needed
    if 'AC_Power' in std_df.columns and std_df['AC_Power'].mean() > 100:  # Likely in watts
        print("Converting AC_Power from watts to kilowatts")
        std_df['AC_Power'] = std_df['AC_Power'] / 1000
    
    if 'DC_Power' in std_df.columns and std_df['DC_Power'].mean() > 100:  # Likely in watts
        print("Converting DC_Power from watts to kilowatts")
        std_df['DC_Power'] = std_df['DC_Power'] / 1000
    
    # Ensure Efficiency is in percentage
    if 'Efficiency' in std_df.columns and std_df['Efficiency'].max() <= 1:  # Likely in decimal
        print("Converting Efficiency from decimal to percentage")
        std_df['Efficiency'] = std_df['Efficiency'] * 100
    
    # Map specific inverter data format to standard fields
    field_mapping = {
        # Format: 'input_column': 'standard_column'
        'Time': 'Timestamp',
        'Pac(W)': 'AC_Power',
        'Ppv(W)': 'DC_Power',
        'VacR(V)': 'Voltage_AC',  # Using R phase voltage
        'IacR(A)': 'Current_AC',  # Using R phase current
        'Fac(Hz)': 'Frequency'
    }
    
    # Check if this is the inverter format and apply mapping
    inverter_keys = ['Serial number', 'Time', 'Pac(W)', 'Ppv(W)', 'Fac(Hz)']
    if all(key in df.columns for key in inverter_keys):
        print("Detected inverter data format - mapping fields to standard format")
        # Create new dataframe with mapped columns
        mapped_df = pd.DataFrame()
        
        for input_col, std_col in field_mapping.items():
            if input_col in df.columns:
                mapped_df[std_col] = df[input_col]
                print(f"Mapped {input_col} → {std_col}")
                
                # Convert power columns from watts to kilowatts
                if std_col in ['AC_Power', 'DC_Power'] and mapped_df[std_col].mean() > 100:
                    mapped_df[std_col] = mapped_df[std_col] / 1000
                    print(f"Converting {std_col} from watts to kilowatts")
        
        # Convert timestamp to datetime if it's not already
        if 'Timestamp' in mapped_df.columns and not pd.api.types.is_datetime64_any_dtype(mapped_df['Timestamp']):
            mapped_df['Timestamp'] = pd.to_datetime(mapped_df['Timestamp'])
            print("Converted Timestamp to datetime format")
            
        # Use the mapped dataframe instead
        std_df = mapped_df
    
    # Add any missing required columns
    required_columns = [
        'Timestamp', 'AC_Power', 'DC_Power', 'Voltage_AC', 
        'Current_AC', 'Frequency', 'Efficiency', 'Status_Flags'
    ]
    
    # Add timestamp timezone info if not present
    if 'Timestamp' in std_df.columns and pd.api.types.is_datetime64_any_dtype(std_df['Timestamp']) and std_df['Timestamp'].dt.tz is None:
        # Assuming local time if not specified
        print("Adding timezone info to Timestamp (assuming local time)")
        try:
            std_df['Timestamp'] = std_df['Timestamp'].dt.tz_localize('local')
        except Exception as e:
            print(f"Could not localize timezone: {e}")
    
    # Separate Timestamp into Date and Time columns
    if 'Timestamp' in std_df.columns and pd.api.types.is_datetime64_any_dtype(std_df['Timestamp']):
        print("Separating Timestamp into Date and Time columns")
        std_df['Date'] = std_df['Timestamp'].dt.date
        std_df['Time'] = std_df['Timestamp'].dt.time
        
        # Reorder columns to place Date and Time right after Timestamp
        cols = list(std_df.columns)
        # Remove Date and Time from their current positions
        cols.remove('Date')
        cols.remove('Time')
        # Find the position of Timestamp
        timestamp_idx = cols.index('Timestamp')
        # Insert Date and Time after Timestamp
        cols.insert(timestamp_idx + 1, 'Date')
        cols.insert(timestamp_idx + 2, 'Time')
        # Reorder the DataFrame
        std_df = std_df[cols]
        
        print("Created Date and Time columns as second and third columns")
    
    # Add descriptive metadata
    metadata = {
        'Timestamp': 'Time of measurement (UTC/local)',
        'Date': 'Date of measurement (YYYY-MM-DD)',
        'Time': 'Time of measurement (HH:MM:SS)',
        'AC_Power': 'Active power output (kW)',
        'DC_Power': 'DC power from panels (kW)',
        'Voltage_AC': 'Grid voltage (V)',
        'Current_AC': 'Grid current (A)',
        'Frequency': 'Grid frequency (Hz)',
        'Efficiency': 'Inverter efficiency (%)',
        'Status_Flags': 'Error/status codes'
    }
    
    # Update required columns to include Date and Time
    required_columns.extend(['Date', 'Time'])
    
    # Add any missing columns as NaN
    for col in required_columns:
        if col not in std_df.columns:
            std_df[col] = np.nan
            print(f"Added missing column: {col}")
    
    print("Standardization complete")
    return std_df

def process_solar_data_file(input_file, output_file=None, output_format='csv'):
    """
    Process a solar data file to standardize fields and units
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path where the standardized file should be saved
    output_format : str, optional
        File format to save ('csv', 'excel', 'pickle', 'parquet')
        
    Returns:
    --------
    str
        Path to the saved standardized file
    """
    # Import the data
    df = import_solar_data(input_file)
    
    # Standardize the data
    std_df = standardize_solar_data(df)
    
    # Analyze the data (without saving)
    analyze_solar_data(std_df)
    
    # Save the standardized data
    saved_path = save_solar_data(std_df, output_path=output_file, format=output_format)
    
    return saved_path

def process_multiple_solar_files(input_dir, output_format='csv'):
    """
    Process multiple solar data files from a directory
    
    Parameters:
    -----------
    input_dir : str
        Path to the directory containing solar data files
    output_format : str, optional
        File format to save ('csv', 'excel', 'pickle', 'parquet')
        
    Returns:
    --------
    list
        List of paths to the saved standardized files
    """
    # Ensure path ends with slash
    if not input_dir.endswith('/'):
        input_dir += '/'
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Get all CSV files in the directory
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                 if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return []
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_files = []
    errors = []
    
    # Process each file
    for i, file_path in enumerate(csv_files, 1):
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"processed_{file_name.split('.')[0]}")
        
        try:
            print(f"Processing file {i}/{len(csv_files)}: {file_name}")
            output_path = process_solar_data_file(
                input_file=file_path,
                output_file=output_file,
                output_format=output_format
            )
            processed_files.append(output_path)
            print(f"Successfully processed: {file_name}")
            
        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total files: {len(csv_files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed: {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"- {err}")
    
    return processed_files

# Example usage
if __name__ == "__main__":
    # Directory containing inverter data files
    inverter_data_dir = "./inverters/"
    output_format = 'csv'  # can be 'csv', 'excel', 'pickle', 'parquet'
    
    try:
        # Process all files in the directory
        processed_files = process_multiple_solar_files(
            input_dir=inverter_data_dir,
            output_format=output_format
        )
        
        if processed_files:
            print(f"\nAll processed files saved to: {os.path.dirname(processed_files[0])}")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
