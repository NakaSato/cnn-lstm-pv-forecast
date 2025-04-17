import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clean_inverter_data(file_path, output_folder):

  # Create output filename
  filename = os.path.basename(file_path)
  output_path = os.path.join(output_folder, filename.replace('.csv', '_cleaned.csv'))
  
  # Read the file content
  with open(file_path, 'r') as f:
    content = f.read()
  
  # Find the actual data header line (the one that starts with "Serial number,Time,Status,...")
  data_header_match = re.search(r'Serial number,Time,Status.*', content)
  if data_header_match:
    data_header_pos = data_header_match.start()
    data_section = content[data_header_pos:]
    
    # Parse the data section
    try:
      # Write the data section to a temporary file
      temp_file = file_path + '.temp'
      with open(temp_file, 'w') as f:
        f.write(data_section)
      
      # Read the data using pandas
      df = pd.read_csv(temp_file)
      
      # Clean the data
      # 1. Remove rows with all NaN values
      df = df.dropna(how='all')
      
      # 2. Convert date/time columns to datetime format
      if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
      if 'lastUpdateTime' in df.columns:
        df['lastUpdateTime'] = pd.to_datetime(df['lastUpdateTime'], errors='coerce')
      
      # 3. Convert numeric columns to appropriate types
      numeric_cols = df.select_dtypes(include=['float64']).columns
      for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
      # Save cleaned data
      df.to_csv(output_path, index=False)
      
      # Clean up temporary file
      if os.path.exists(temp_file):
        os.remove(temp_file)
        
      print(f"Cleaned data saved to {output_path}")
      return df
      
    except Exception as e:
      print(f"Error processing data: {e}")
      return None
  else:
    print(f"Could not find data header in the file: {file_path}")
    return None

def plot_inverter_data(df, output_folder, filename):

  if df is None or df.empty:
    print("No data to plot")
    return
  
  # Create plots folder if it doesn't exist
  plots_folder = os.path.join(output_folder, 'plots')
  os.makedirs(plots_folder, exist_ok=True)
  
  # Create a specific folder for this file's plots
  file_plots_folder = os.path.join(plots_folder, filename.replace('.csv', ''))
  os.makedirs(file_plots_folder, exist_ok=True)
  
  try:
    # Enhanced styling
    plt.rcParams.update({'font.size': 12})
    sns.set_style("whitegrid")
    colors = plt.cm.tab10.colors
    
    # Plot time series for key metrics with improved styling
    if 'Time' in df.columns:
      numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
      
      # Time series plots
      for i, col in enumerate(numeric_cols[:min(8, len(numeric_cols))]):
        plt.figure(figsize=(14, 7))
        
        # Plot with better styling
        plt.plot(df['Time'], df[col], '-o', color=colors[i % len(colors)], 
                 linewidth=2, markersize=5, alpha=0.8)
        
        # Add moving average for trend visualization
        if len(df) > 5:  # Only add if we have enough data points
          window_size = max(3, len(df) // 10)  # Dynamic window size
          # Create temporary DataFrame for moving average calculation
          temp_df = pd.DataFrame({'Time': df['Time'], 'value': df[col], 
                                 'moving_avg': df[col].rolling(window=window_size).mean()})
          plt.plot(temp_df['Time'], temp_df['moving_avg'], 'r--', 
                   linewidth=2, label=f'Moving Avg ({window_size} points)')
          plt.legend()
        
        # Improved title and labels
        plt.title(f'{col} Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(col, fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add data statistics as text
        stats = f"Min: {df[col].min():.2f}\nMax: {df[col].max():.2f}\nMean: {df[col].mean():.2f}"
        plt.figtext(0.02, 0.02, stats, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plot_path = os.path.join(file_plots_folder, f"{col}_time_series.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Time series plot saved to {plot_path}")
      
      # Add distribution plots (histograms)
      for i, col in enumerate(numeric_cols[:min(8, len(numeric_cols))]):
        plt.figure(figsize=(12, 6))
        sns.histplot(df[col], kde=True, color=colors[i % len(colors)], bins=25)
        plt.title(f'Distribution of {col}', fontsize=16, fontweight='bold')
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(file_plots_folder, f"{col}_histogram.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Histogram saved to {plot_path}")
      
      # Correlation heatmap (if we have multiple numeric columns)
      if len(numeric_cols) > 1:
        plt.figure(figsize=(max(10, len(numeric_cols) * 1.5), max(8, len(numeric_cols) * 1.2)))
        
        # Calculate and plot correlation matrix
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
        
        heatmap = sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            linewidths=0.5,
            vmin=-1, 
            vmax=1,
            center=0
        )
        
        plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(file_plots_folder, "correlation_heatmap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Correlation heatmap saved to {plot_path}")
      
      # Box plots for comparing distributions
      if len(numeric_cols) >= 2:
        plt.figure(figsize=(max(10, len(numeric_cols) * 1.2), 8))
        
        # Select columns to plot (limit to prevent overcrowding)
        plot_cols = numeric_cols[:min(10, len(numeric_cols))]
        
        # Normalize data for meaningful comparison
        normalized_data = df[plot_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x)
        
        # Create boxplot
        sns.boxplot(data=normalized_data, palette='Set2')
        plt.title('Normalized Distribution Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Normalized Values', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(file_plots_folder, "boxplot_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Box plot comparison saved to {plot_path}")
      
      # Create time-based aggregation if enough data
      if len(df) > 10 and 'Time' in df.columns:
        # Add time-based columns
        df['Hour'] = df['Time'].dt.hour
        df['Date'] = df['Time'].dt.date
        
        # If data spans multiple days, create daily aggregation
        if df['Date'].nunique() > 1:
          for i, col in enumerate(numeric_cols[:min(5, len(numeric_cols))]):
            plt.figure(figsize=(14, 7))
            
            # Group by date and calculate statistics
            daily_stats = df.groupby('Date')[col].agg(['mean', 'min', 'max']).reset_index()
            
            # Plot daily statistics
            plt.fill_between(daily_stats['Date'], daily_stats['min'], daily_stats['max'], 
                            alpha=0.3, color=colors[i % len(colors)])
            plt.plot(daily_stats['Date'], daily_stats['mean'], 'o-', 
                    color=colors[i % len(colors)], linewidth=2, markersize=6)
            
            plt.title(f'Daily {col} Statistics', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel(col, fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_path = os.path.join(file_plots_folder, f"{col}_daily_aggregation.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Daily aggregation plot saved to {plot_path}")
      
      # Create a dashboard with multiple plots if we have enough data
      if len(numeric_cols) >= 2:
        plt.figure(figsize=(16, 12))
        plt.suptitle(f'Dashboard for {filename}', fontsize=20, fontweight='bold', y=0.95)
        
        # Select top metrics to display
        top_cols = numeric_cols[:min(2, len(numeric_cols))]
        
        # Create a 2x2 subplot grid
        gs = plt.GridSpec(2, 2)
        
        # Time series for first metric
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(df['Time'], df[top_cols[0]], '-o', linewidth=2, markersize=4, alpha=0.8)
        ax1.set_title(f'{top_cols[0]} Over Time', fontsize=14)
        ax1.set_xlabel('Time')
        ax1.set_ylabel(top_cols[0])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Histogram for first metric
        ax2 = plt.subplot(gs[0, 1])
        sns.histplot(df[top_cols[0]], kde=True, ax=ax2, bins=20)
        ax2.set_title(f'Distribution of {top_cols[0]}', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # If we have a second column, show its time series
        if len(top_cols) > 1:
            ax3 = plt.subplot(gs[1, 0])
            ax3.plot(df['Time'], df[top_cols[1]], '-o', linewidth=2, markersize=4, 
                     alpha=0.8, color='green')
            ax3.set_title(f'{top_cols[1]} Over Time', fontsize=14)
            ax3.set_xlabel('Time')
            ax3.set_ylabel(top_cols[1])
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Scatter plot showing relationship between two metrics
            ax4 = plt.subplot(gs[1, 1])
            ax4.scatter(df[top_cols[0]], df[top_cols[1]], alpha=0.7)
            ax4.set_title(f'{top_cols[0]} vs {top_cols[1]}', fontsize=14)
            ax4.set_xlabel(top_cols[0])
            ax4.set_ylabel(top_cols[1])
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Add correlation coefficient as text
            corr_val = df[top_cols].corr().iloc[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr_val:.2f}', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        plot_path = os.path.join(file_plots_folder, "dashboard.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Dashboard saved to {plot_path}")
        
  except Exception as e:
    print(f"Error creating plots: {e}")
    import traceback
    print(traceback.format_exc())

def merge_all_cleaned_files(cleaned_folder, output_file):

  # Find all cleaned CSV files
  cleaned_files = glob.glob(os.path.join(cleaned_folder, '*_cleaned.csv'))
  
  if not cleaned_files:
    print(f"No cleaned CSV files found in {cleaned_folder}")
    return None
  
  print(f"Found {len(cleaned_files)} cleaned files to merge")
  
  # Initialize an empty list to store dataframes
  dfs = []
  
  # Read and combine all files
  for file in cleaned_files:
    try:
      df = pd.read_csv(file)
      # Add filename as source column
      df['source_file'] = os.path.basename(file)
      dfs.append(df)
      print(f"Added {file} to merged data")
    except Exception as e:
      print(f"Error reading {file}: {e}")
  
  if not dfs:
    print("No valid data files to merge")
    return None
  
  # Concatenate all dataframes
  merged_df = pd.concat(dfs, ignore_index=True)
  
  # Save the merged file
  merged_df.to_csv(output_file, index=False)
  print(f"Merged data saved to {output_file}")
  
  return merged_df

# Execute the cleaning function
if __name__ == "__main__":
  input_folder = "./inverter/"
  
  # Check if the folder exists
  if not os.path.isdir(input_folder):
    print(f"The directory {input_folder} does not exist.")
    input_folder = input("Please enter the path to the folder containing CSV files: ")
  
  # Create output folders
  cleaned_folder = "./cleaned/"
  os.makedirs(cleaned_folder, exist_ok=True)
  
  # Find all CSV files in the folder
  csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
  
  if not csv_files:
    print(f"No CSV files found in {input_folder}")
  else:
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_dfs = []
    for file_path in csv_files:
      print(f"\nProcessing: {file_path}")
      filename = os.path.basename(file_path)
      df = clean_inverter_data(file_path, cleaned_folder)
      if df is not None:
        plot_inverter_data(df, cleaned_folder, filename)
    
    # Merge all cleaned files
    merged_output = os.path.join(cleaned_folder, "all_inverters_merged.csv")
    merge_all_cleaned_files(cleaned_folder, merged_output)
