import os
from size import run_ga_for_season
from models.load_data import load_data

def main():
    # Load preprocessed seasonal data
    project_root = os.path.dirname(__file__)
    data_path = os.path.join(project_root, "models", "data", "Accepted_Week_Data.xlsx")
    seasonal_data = load_data(file_path=data_path)

    if seasonal_data is None:
        print("Error: Could not load data.")
        return

    summary = {}

    for season_name, season_df in seasonal_data.items():
        print(f"\n--- Optimizing for {season_name} ---")
        result = run_ga_for_season(season_df)

        summary[season_name] = {
            "PV": int(result["best_pv"]),
            "Wind": int(result["best_wind"]),
            "Batteries": int(result["best_battery"]),
            "DG": int(result["best_dg"])
        }

    # Final summary
    print("\n--- Final Summary Across All Seasons ---")
    for season, config in summary.items():
        print(f"{season}: PV = {config['PV']}, Wind = {config['Wind']}, Batteries = {config['Batteries']}, DG = {config['DG']}")

if __name__ == "__main__":
    main()
