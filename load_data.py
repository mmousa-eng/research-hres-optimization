import pandas as pd

def load_data(file_path="data/Accepted_Week_Data.xlsx"):
    """
    Loads the HRES data from the specified Excel file, preprocesses it,
    and segments it into seasonal DataFrames, accounting for 'Unnamed' columns.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        dict: A dictionary where keys are season names (str) and values
              are pandas DataFrames containing the preprocessed data for that season.
    """
    try:
        # Load the Excel file, using the 2nd row as headers (header=1)
        df_raw = pd.read_excel(file_path, sheet_name="Week Data", header=1)
        print(f"Successfully loaded '{file_path}'.")
        print(f"Raw DataFrame shape: {df_raw.shape}")

        # Define the standardized column names we want for each seasonal DataFrame
        column_names = [
            "Year", "Month", "Day", "Hour", "G(W/m^2)", "Ta(C) at 2m",
            "V(m/s) at 10m", "Ta(k) at 2m", "Load (kW)"
        ]

        # Explicitly define the starting column index for each season based on your Excel structure
        # (9 actual data columns + 1 'Unnamed' column = 10 columns per block in the raw data)
        # Winter: starts at index 0
        # Spring: starts at index 10 (after 'Unnamed: 9')
        # Summer: starts at index 20 (after 'Unnamed: 19')
        # Fall:   starts at index 30 (after 'Unnamed: 29')
        season_start_cols = {
            "Winter": 0,
            "Spring": 10,
            "Summer": 20,
            "Fall": 30
        }

        # Number of rows per season confirmed to be 182, but 2 are blank, resulting in 180 actual data rows.
        num_rows_to_read = 182
        num_columns_per_season_data = len(column_names) # 9 actual data columns

        seasonal_data = {}
        season_names = ["Winter", "Spring", "Summer", "Fall"]

        for season_name in season_names:
            start_col_idx = season_start_cols[season_name]
            end_col_idx = start_col_idx + num_columns_per_season_data

            # Select the relevant rows and columns for the current season
            season_df = df_raw.iloc[:num_rows_to_read, start_col_idx:end_col_idx].copy()
            season_df.columns = column_names # Assign standardized column names

            # Basic cleaning: drop any rows that might be entirely empty
            season_df.dropna(how='all', inplace=True)

            # Ensure 'Hour' column is 0-indexed (0 to 23) if it's 1-indexed (1 to 24)
            # This check prevents issues if some 'Hour' columns are already 0-indexed.
            if not season_df.empty and season_df['Hour'].max() == 24:
                season_df['Hour'] = season_df['Hour'] - 1

            # Convert relevant columns to numeric, coercing errors to NaN and then dropping rows with NaNs in critical columns
            for col in ["G(W/m^2)", "Ta(C) at 2m", "V(m/s) at 10m", "Ta(k) at 2m", "Load (kW)"]:
                season_df[col] = pd.to_numeric(season_df[col], errors='coerce')
            season_df.dropna(subset=["G(W/m^2)", "V(m/s) at 10m", "Load (kW)"], inplace=True)

            seasonal_data[season_name] = season_df
            print(f"  Processed {season_name} data. Shape: {season_df.shape}")

        return seasonal_data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the 'data/' folder.")
        return None
    except KeyError as e:
        print(f"Error processing Excel sheet or columns: {e}. Check sheet name 'Week Data' and column structure.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

# This part is for testing the function if you run data_preprocessing.py directly
if __name__ == "__main__":
    processed_data = load_data()

    if processed_data:
        print("\nData loaded and preprocessed successfully for all seasons:")
        for season, df in processed_data.items():
            print(f"\n--- {season} Data (first 5 rows) ---")
            print(df.head())
            print(f"  Shape: {df.shape}")