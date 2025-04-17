import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns

def import_solar_data(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    # Read the CSV file
    print(f"Importing data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Enhanced column mapping with multiple possible input formats
    column_mapping = {
        # Serial Number variations
        'Serial number': 'Serial_Number',
        'Serial_number': 'Serial_Number',
        'SerialNumber': 'Serial_Number',
        'SN': 'Serial_Number',
        'Device ID': 'Serial_Number',
        
        # Timestamp variations
        'Time': 'Timestamp',
        'DateTime': 'Timestamp',
        'Date Time': 'Timestamp',
        'Date_Time': 'Timestamp',
        'Timestamp': 'Timestamp',
        'Date': 'Timestamp',
        
        # AC Power variations
        'Pac(W)': 'AC_Power',
        'Pac': 'AC_Power',
        'AC Power': 'AC_Power',
        'AC Power(W)': 'AC_Power',
        'AC_Power(W)': 'AC_Power',
        'Power AC': 'AC_Power',
        
        # DC Power variations
        'Ppv(W)': 'DC_Power',
        'Ppv': 'DC_Power',
        'DC Power': 'DC_Power',
        'DC Power(W)': 'DC_Power',
        'DC_Power(W)': 'DC_Power',
        'Power DC': 'DC_Power',
        
        # Voltage AC variations - R phase
        'VacR(V)': 'Voltage_AC_R',
        'Vac_R': 'Voltage_AC_R',
        'V-R': 'Voltage_AC_R',
        
        # Voltage AC variations - S phase
        'VacS(V)': 'Voltage_AC_S',
        'Vac_S': 'Voltage_AC_S',
        'V-S': 'Voltage_AC_S',
        
        # Voltage AC variations - T phase
        'VacT(V)': 'Voltage_AC_T',
        'Vac_T': 'Voltage_AC_T',
        'V-T': 'Voltage_AC_T',
        
        # Voltage AC variations - phase to phase
        'VacRS(V)': 'Voltage_AC_RS',
        'Vac_RS': 'Voltage_AC_RS',
        'V-RS': 'Voltage_AC_RS',
        
        'VacST(V)': 'Voltage_AC_ST',
        'Vac_ST': 'Voltage_AC_ST',
        'V-ST': 'Voltage_AC_ST',
        
        'VacTR(V)': 'Voltage_AC_TR',
        'Vac_TR': 'Voltage_AC_TR',
        'V-TR': 'Voltage_AC_TR',
        
        # Current AC variations - R phase
        'IacR(A)': 'Current_AC_R',
        'Iac_R': 'Current_AC_R',
        'I-R': 'Current_AC_R',
        
        # Current AC variations - S phase
        'IacS(A)': 'Current_AC_S',
        'Iac_S': 'Current_AC_S',
        'I-S': 'Current_AC_S',
        
        # Current AC variations - T phase
        'IacT(A)': 'Current_AC_T',
        'Iac_T': 'Current_AC_T',
        'I-T': 'Current_AC_T',
        
        # Frequency variations
        'Fac(Hz)': 'Frequency',
        'Fac': 'Frequency',
        'Frequency(Hz)': 'Frequency',
        'F(Hz)': 'Frequency',
        'Grid Frequency': 'Frequency'
    }
    
    # Create case-insensitive mapping for better matching
    case_insensitive_mapping = {}
    for src, dst in column_mapping.items():
        case_insensitive_mapping[src.lower()] = dst
    
    # Apply column mapping with better tracking
    mapped_columns = []
    lower_columns = {col.lower(): col for col in df.columns}
    
    for src_lower, dst in case_insensitive_mapping.items():
        if src_lower in lower_columns:
            original_col = lower_columns[src_lower]
            df[dst] = df[original_col]
            mapped_columns.append((original_col, dst))
    
    if mapped_columns:
        print(f"Mapped {len(mapped_columns)} columns:")
        for src, dst in mapped_columns:
            print(f"  • {src} → {dst}")
    else:
        print("Warning: No columns were mapped from the expected format")
            
    # Ensure Serial_Number exists and convert to string type
    if 'Serial_Number' in df.columns:
        df['Serial_Number'] = df['Serial_Number'].astype(str)
        print(f"Found Serial_Number field with {df['Serial_Number'].nunique()} unique values")
    else:
        print("Warning: Serial_Number field not found in data")
    
    # Calculate average Voltage_AC and Current_AC if 3-phase data is available
    voltage_cols = ['Voltage_AC_R', 'Voltage_AC_S', 'Voltage_AC_T']
    current_cols = ['Current_AC_R', 'Current_AC_S', 'Current_AC_T']
    
    # Use available voltage columns with vectorized operations instead of checking all columns
    avail_voltage_cols = [col for col in voltage_cols if col in df.columns]
    if avail_voltage_cols:
        df['Voltage_AC'] = df[avail_voltage_cols].mean(axis=1)
        print(f"Calculated Voltage_AC from {len(avail_voltage_cols)} phases")
    
    # Use available current columns with vectorized operations
    avail_current_cols = [col for col in current_cols if col in df.columns]
    if avail_current_cols:
        df['Current_AC'] = df[avail_current_cols].mean(axis=1)
        print(f"Calculated Current_AC from {len(avail_current_cols)} phases")
    
    # Calculate efficiency with vectorized operation and proper error handling
    if 'Efficiency' not in df.columns and 'AC_Power' in df.columns and 'DC_Power' in df.columns:
        # Avoid division by zero and handle edge cases with numpy where
        valid_dc = df['DC_Power'] > 10  # Threshold to avoid division by very small values
        df['Efficiency'] = np.where(valid_dc, 
                                   np.clip(df['AC_Power'] / df['DC_Power'] * 100, 0, 100), 
                                   0)
        
        # Flag unrealistic efficiency values (above theoretical limits)
        unrealistic = (df['Efficiency'] > 98) & valid_dc
        if unrealistic.any():
            print(f"Warning: {unrealistic.sum()} records have suspiciously high efficiency (>98%)")
    
    # Check if essential columns exist
    essential_columns = ['Timestamp', 'AC_Power', 'DC_Power']
    missing_essential = [col for col in essential_columns if col not in df.columns]
    if missing_essential:
        raise ValueError(f"Missing essential columns: {missing_essential}. Cannot proceed with analysis.")
    
    # Define expected columns for logging purposes
    expected_columns = [
        'Serial_Number', 'Timestamp', 'AC_Power', 'DC_Power',
        'Voltage_AC', 'Current_AC', 'Frequency', 'Efficiency'
    ]
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing expected columns: {missing_columns}")
    
    # Convert timestamp to datetime with robust error handling
    if 'Timestamp' in df.columns:
        # Try common datetime formats in order of likelihood
        timestamp_formats = [
            None,  # Let pandas infer the format
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S'
        ]
        
        for fmt in timestamp_formats:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt)
                print(f"Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
                break
            except Exception:
                continue
        else:
            print("Error: Could not parse Timestamp column with any known format")
    
    # Handle missing values more efficiently with vectorized operations
    numeric_cols = [col for col in df.columns if df[col].dtype.kind in 'ifc']  # integer, float, complex
    missing_counts = df[numeric_cols].isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if not cols_with_missing.empty:
        print("Handling missing values:")
        for col, count in cols_with_missing.items():
            percent_missing = (count / len(df)) * 100
            print(f"  {col}: {count} missing values ({percent_missing:.2f}%)")
            
            # Handle based on importance and missing percentage
            if col in ['AC_Power', 'DC_Power']:
                if percent_missing < 5:  # Less than 5% missing
                    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                else:
                    print(f"  Warning: High percentage of missing values in {col}")
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Check for outliers in key metrics using IQR method
    outlier_cols = ['AC_Power', 'DC_Power', 'Efficiency'] if 'Efficiency' in df.columns else ['AC_Power', 'DC_Power']
    outlier_summary = {}
    
    for col in outlier_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                outlier_summary[col] = len(outliers)
                # Mark outliers for reference but don't modify them
                df[f'{col}_is_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    if outlier_summary:
        print("Potential outliers detected:")
        for col, count in outlier_summary.items():
            print(f"  {col}: {count} outliers ({(count/len(df)*100):.2f}%)")
    
    # Add data quality score
    quality_metrics = []
    if 'Efficiency' in df.columns:
        quality_metrics.append(((df['Efficiency'] <= 100) & (df['Efficiency'] > 0)).mean())
    if 'AC_Power' in df.columns and 'DC_Power' in df.columns:
        quality_metrics.append((df['AC_Power'] <= df['DC_Power']).mean())
    
    if quality_metrics:
        df_quality = sum(quality_metrics) / len(quality_metrics)
        print(f"Data quality score: {df_quality:.2%}")
    
    print(f"Successfully imported data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def analyze_solar_data(df, save_plots=True, plot_dir='./plots_dataset', show_plots=False):

    print("\nBasic Statistics:")
    # Only analyze numeric columns for statistics
    numeric_df = df.select_dtypes(include=['number'])
    print(numeric_df.describe())
    
    # Create plot directory if needed
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created plot directory: {plot_dir}")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot configuration for consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'tertiary': '#2ca02c', 'highlight': '#d62728'}
    
    # Check data intervals if timestamp exists
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp')
        time_diffs = df['Timestamp'].diff().dropna()
        if not time_diffs.empty:
            common_interval = time_diffs.value_counts().idxmax()
            median_interval = time_diffs.median()
            print(f"\nMost common time interval: {common_interval}")
            print(f"Median time interval: {median_interval}")
            
            # Check for data gaps
            large_gaps = time_diffs[time_diffs > 2 * median_interval]
            if not large_gaps.empty:
                print(f"Found {len(large_gaps)} large time gaps. Largest gap: {large_gaps.max()}")
    
    # Plot AC vs DC power with better visualization
    if 'AC_Power' in df.columns and 'DC_Power' in df.columns:
        valid_data = df.dropna(subset=['AC_Power', 'DC_Power'])
        
        plt.figure(figsize=(12, 8))
        
        # Use hexbin for large datasets to avoid overplotting
        if len(valid_data) > 5000:
            plt.hexbin(valid_data['DC_Power'], valid_data['AC_Power'], 
                      gridsize=50, cmap='Blues', bins='log')
            plt.colorbar(label='Log10(count)')
        else:
            plt.scatter(valid_data['DC_Power'], valid_data['AC_Power'], 
                       alpha=0.6, color=colors['primary'], edgecolor='k', linewidth=0.5)
        
        # Add regression line with confidence interval
        try:
            import scipy.stats as stats
            
            # Linear regression
            mask = ~np.isnan(valid_data['DC_Power']) & ~np.isnan(valid_data['AC_Power'])
            x = valid_data['DC_Power'][mask]
            y = valid_data['AC_Power'][mask]
            
            if len(x) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_range = np.linspace(x.min(), x.max(), 100)
                plt.plot(x_range, slope*x_range + intercept, 'r--', linewidth=2, 
                        label=f'Trend: y = {slope:.4f}x + {intercept:.2f}\nR² = {r_value**2:.4f}')
                
                # Calculate correlation
                print(f"AC-DC Power Correlation: {r_value:.4f} (R² = {r_value**2:.4f})")
        except ImportError:
            # Fallback to numpy if scipy not available
            z = np.polyfit(valid_data['DC_Power'], valid_data['AC_Power'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(valid_data['DC_Power'].min(), valid_data['AC_Power'].max(), 100)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2, 
                    label=f'Trend: y = {z[0]:.4f}x + {z[1]:.2f}')
        
        plt.xlabel('DC Power (W)', fontsize=12, fontweight='bold')
        plt.ylabel('AC Power (W)', fontsize=12, fontweight='bold')
        plt.title('AC Power vs DC Power Relationship', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(plot_dir, f"ac_vs_dc_power_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved AC vs DC Power plot to {plot_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Calculate and analyze efficiency
    efficiency_col = None
    if 'Efficiency' in df.columns:
        efficiency_col = 'Efficiency'
    elif 'AC_Power' in df.columns and 'DC_Power' in df.columns:
        # Calculate efficiency with proper handling of edge cases
        df['Calculated_Efficiency'] = np.where(
            df['DC_Power'] > 10,  # Only calculate when DC power is significant
            np.clip(df['AC_Power'] / df['DC_Power'] * 100, 0, 100),
            np.nan
        )
        efficiency_col = 'Calculated_Efficiency'
        
    if efficiency_col:
        valid_eff = df[efficiency_col].dropna()
        
        if not valid_eff.empty:
            print(f"\n{efficiency_col} statistics:")
            stats_to_display = valid_eff.describe(percentiles=[.05, .25, .5, .75, .95])
            print(stats_to_display)
            
            # Create more informative efficiency histogram
            plt.figure(figsize=(12, 7))
            
            # Use KDE for smoother visualization
            n, bins, patches = plt.hist(valid_eff, bins=30, color=colors['tertiary'], 
                                       alpha=0.7, edgecolor='black', density=True)
            
            # Add percentile lines
            percentiles = [5, 25, 50, 75, 95]
            colors_perc = ['#8B0000', '#FF4500', '#000000', '#228B22', '#006400']
            labels_perc = ['5th', '25th', '50th\n(Median)', '75th', '95th']
            
            for i, p in enumerate(percentiles):
                perc_val = valid_eff.quantile(p/100)
                plt.axvline(perc_val, color=colors_perc[i], linestyle='--', linewidth=2, 
                           label=f"{labels_perc[i]}: {perc_val:.2f}%")
            
            # Add mean line
            plt.axvline(valid_eff.mean(), color='blue', linestyle='-', linewidth=2.5,
                       label=f'Mean: {valid_eff.mean():.2f}%')
            
            plt.xlabel('Efficiency (%)', fontsize=12, fontweight='bold')
            plt.ylabel('Density', fontsize=12, fontweight='bold')
            plt.title('Distribution of Inverter Efficiency', fontsize=14, fontweight='bold')
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if save_plots:
                plot_file = os.path.join(plot_dir, f"efficiency_histogram_{timestamp}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Saved Efficiency Histogram to {plot_file}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # Enhanced daily power generation pattern with shaded error bars
    if 'Timestamp' in df.columns and 'AC_Power' in df.columns:
        # Add hour and date for aggregation
        df['Hour'] = df['Timestamp'].dt.hour
        df['Date'] = df['Timestamp'].dt.date
        
        # Calculate hourly statistics across all days
        hourly_stats = df.groupby('Hour')['AC_Power'].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Calculate standard error for error bars
        hourly_stats['se'] = hourly_stats['std'] / np.sqrt(hourly_stats['count'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot mean with error bars
        ax.bar(hourly_stats.index, hourly_stats['mean'], 
              yerr=hourly_stats['se'], 
              color=colors['secondary'], edgecolor='black', 
              capsize=5, alpha=0.7, width=0.7)
        
        # Add line showing the trend
        ax.plot(hourly_stats.index, hourly_stats['mean'], 'o-', 
               color='darkred', linewidth=2, markersize=8)
        
        # Add min-max range as a shaded area
        ax.fill_between(hourly_stats.index, 
                       hourly_stats['min'], 
                       hourly_stats['max'], 
                       alpha=0.2, color=colors['primary'],
                       label='Min-Max Range')
        
        # Format the plot
        hour_labels = [f"{h:02d}:00" for h in hourly_stats.index]
        ax.set_xticks(range(len(hour_labels)))
        ax.set_xticklabels(hour_labels, rotation=45)
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('AC Power (W)', fontsize=12, fontweight='bold')
        ax.set_title('Average Power Output Throughout the Day', fontsize=14, fontweight='bold')
        
        # Show the number of data points used for each hour
        for i, (_, row) in enumerate(hourly_stats.iterrows()):
            ax.annotate(f"n={int(row['count'])}", 
                       (i, row['mean'] + row['se'] + (hourly_stats['mean'].max() * 0.02)),
                       ha='center', fontsize=8)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(plot_dir, f"hourly_power_output_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved Hourly Power Output plot to {plot_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Add daily power generation heatmap if multiple days exist
        if df['Date'].nunique() > 1:
            # Pivot data for heatmap: days vs hours
            daily_hourly = df.pivot_table(
                index=df['Timestamp'].dt.date,
                columns='Hour',
                values='AC_Power',
                aggfunc='mean'
            )
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            
            # Check if seaborn is available for better heatmap
            try:
                ax = sns.heatmap(daily_hourly, cmap='viridis', 
                               linewidths=0.5, 
                               cbar_kws={'label': 'Average AC Power (W)'})
            except ImportError:
                # Fallback to matplotlib
                ax = plt.matshow(daily_hourly, cmap='viridis', aspect='auto')
                plt.colorbar(label='Average AC Power (W)')
                
            plt.title('Daily Power Generation Pattern', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
            plt.ylabel('Date', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                plot_file = os.path.join(plot_dir, f"daily_power_heatmap_{timestamp}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Saved Daily Power Heatmap to {plot_file}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()

# Example usage
if __name__ == "__main__":
    # Example file path (replace with your actual path)
    file_path = "../raw/cleaned/INVERTER_01_2025-04-04_2025-04-05_cleaned.csv"
    
    try:
        solar_df = import_solar_data(file_path)
        analyze_solar_data(solar_df, save_plots=True, plot_dir='./plots', show_plots=False)
        
        print("\nData preview:")
        print(solar_df.head())
        
    except Exception as e:
        print(f"Error: {e}")