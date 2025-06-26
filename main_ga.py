import os
# Changed from run_ga_for_season to run_annual_ga_optimization
from size_ga import run_annual_ga_optimization 
from utils.load_data import load_data # Assuming this path and loader are correct

def main():
    project_root = os.path.dirname(__file__)
    # Path to data - ensure this is correct for your setup
    data_path = os.path.join(project_root, "models", "data", "Accepted_Week_Data.xlsx")
    
    # seasonal_data is expected to be a dictionary like:
    # {"Winter": df_winter, "Spring": df_spring, ...}
    seasonal_data = load_data(file_path=data_path)

    if not seasonal_data: # Check if dictionary is empty or None
        print("Error: Could not load data or data is empty.")
        return

    print("\n--- Running Annual HRES Optimization ---")
    
    # Call the annual optimization function once, passing all seasonal data
    # The GA in size.py will now iterate through these seasons for each fitness evaluation
    annual_opt_result = run_annual_ga_optimization(seasonal_data)

    # Display the single optimal annual configuration
    print("\n--- Optimal Annual HRES Configuration ---")
    if annual_opt_result and annual_opt_result.get('objective_value') != float('inf'):
        print(f"  Objective Value (Weighted Normalized Cost & Emissions): {annual_opt_result['objective_value']:.4f}")
        print(f"  Optimal Capacities:")
        print(f"    PV: {annual_opt_result['best_pv_kw']:.2f} kW (approx. {annual_opt_result['best_num_pv']} modules)")
        print(f"    Wind: {annual_opt_result['best_wind_kw']:.2f} kW (approx. {annual_opt_result['best_num_wind']} turbines)")
        print(f"    Battery: {annual_opt_result['best_battery_kwh']:.2f} kWh Energy / {annual_opt_result['best_battery_kw_derived']:.2f} kW Power (derived, approx. {annual_opt_result['best_num_battery']} modules for energy)")
        print(f"    DG: {annual_opt_result['best_dg_kw_evaluated']:.2f} kW (evaluated, approx. {annual_opt_result['best_num_dg']} units)")
        
        # To get the unnormalized cost components, you would ideally re-run the fitness function 
        # once with the best capacities and extract those pre-normalization values.
        # This is not done here for simplicity but is good practice for detailed reporting.
        print("\nNote: Objective value is a weighted sum of normalized costs and emissions.")
        print("To see unnormalized TAC and CO2 for the best solution, re-evaluate fitness_function with these capacities.")

    else:
        print("Optimization did not find a valid solution or failed.")

if __name__ == "__main__":
    main()
