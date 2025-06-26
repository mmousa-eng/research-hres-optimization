import os
import numpy as np
import math
from size_nsga2 import run_annual_nsga2_optimization
from utils.load_data import load_data

# Constants for calculating number of units
PV_MODULE_RATING_W = 1000
WIND_TURBINE_RATING_W = 20000
BATTERY_MODULE_ENERGY_WH = 1190
DG_UNIT_RATING_W = 2000

# Imports and other code as before
# ...

def print_solution_summary(solution, seasonal_data, ems_params):
    from ems import estimate_operational_performance
    from models.pv_model import PVModel
    from models.wind_model import WindModel
    from models.batteries_model import BatteryModel
    from models.dg_model import DGModel

    pv_kw = solution['pv_kw']
    wind_kw = solution['wind_kw']
    batt_kwh = solution['battery_kwh']
    dg_kw = solution['dg_kw']

    # Convert to units/modules
    PV_MODULE_RATING_W = 1000
    WIND_TURBINE_RATING_W = 20000
    BATTERY_MODULE_ENERGY_WH = 1190
    DG_UNIT_RATING_W = 2000

    num_pv = max(1, round(pv_kw * 1000 / PV_MODULE_RATING_W))
    num_wind = max(0, round(wind_kw * 1000 / WIND_TURBINE_RATING_W))
    num_batt = max(1, round(batt_kwh * 1000 / BATTERY_MODULE_ENERGY_WH))
    num_dg = max(1, round(dg_kw * 1000 / DG_UNIT_RATING_W))

    # For demonstration, use first season data to estimate performance (or you can average all seasons)
    first_season_name = next(iter(seasonal_data))
    df = seasonal_data[first_season_name]

    pv_model = PVModel(num_modules=num_pv, G=df["G(W/m^2)"].values, Ta=df["Ta(C) at 2m"].values, V=df["V(m/s) at 10m"].values)
    wind_model = WindModel(num_turbines=num_wind, V=df["V(m/s) at 10m"].values)
    batt_model = BatteryModel(num_batteries=num_batt, DT=1, num_steps=len(df))
    dg_model = DGModel(num_generators=num_dg, DT=1, num_steps=len(df))

    pv_power = pv_model.power_output()
    wind_power = wind_model.power_output()

    result = estimate_operational_performance(
        df,
        pv_power,
        wind_power,
        batt_model,
        dg_model,
        ems_params
    )

    dg_contribution_pct = 0.0
    if result['total_load_energy_wh'] > 0:
        dg_contribution_pct = 100.0 * result['total_dg_energy_wh'] / result['total_load_energy_wh']

    print(f"\nSolution Summary:")
    print(f"PV Capacity: {pv_kw:.2f} kW ({num_pv} modules approx.)")
    print(f"Wind Capacity: {wind_kw:.2f} kW ({num_wind} turbines approx.)")
    print(f"Battery Capacity: {batt_kwh:.2f} kWh ({num_batt} modules approx.)")
    print(f"DG Capacity: {dg_kw:.2f} kW ({num_dg} units approx.)")
    print(f"Investment Cost: ${solution['investment_cost']:.2f}")
    print(f"Operational Cost: ${solution['operational_cost']:.2f} / year")
    print(f"CO2 Emissions: {solution['co2_emissions']:.2f} kg / year")
    print(f"DG Contribution to Load (energy basis): {dg_contribution_pct:.2f}%")

# Example usage after you get NSGA-II results:
# Assume 'results' is your list of dicts returned from your optimization
def print_key_solutions(results, seasonal_data, ems_params):
    # Find min investment cost solution
    min_inv_sol = min(results, key=lambda x: x['investment_cost'])
    min_op_sol = min(results, key=lambda x: x['operational_cost'])
    min_co2_sol = min(results, key=lambda x: x['co2_emissions'])
    
    # Normalize criteria for combined (simple equal weighting)
    inv_vals = np.array([r['investment_cost'] for r in results])
    op_vals = np.array([r['operational_cost'] for r in results])
    co2_vals = np.array([r['co2_emissions'] for r in results])

    inv_norm = (inv_vals - inv_vals.min()) / (inv_vals.max() - inv_vals.min() + 1e-9)
    op_norm = (op_vals - op_vals.min()) / (op_vals.max() - op_vals.min() + 1e-9)
    co2_norm = (co2_vals - co2_vals.min()) / (co2_vals.max() - co2_vals.min() + 1e-9)

    combined_scores = inv_norm + op_norm + co2_norm
    best_combined_sol = results[np.argmin(combined_scores)]

    print("\n--- Key Pareto-Optimal HRES Configurations ---")
    print("\n🔹 Minimum Investment Cost Solution")
    print_solution_summary(min_inv_sol, seasonal_data, ems_params)

    print("\n🔹 Minimum Operational Cost Solution")
    print_solution_summary(min_op_sol, seasonal_data, ems_params)

    print("\n🔹 Minimum Emissions Solution")
    print_solution_summary(min_co2_sol, seasonal_data, ems_params)

    print("\n🔹 Best Combined Tradeoff Solution")
    print_solution_summary(best_combined_sol, seasonal_data, ems_params)

# You will call print_key_solutions(results, seasonal_data, ems_params) after running NSGA-II
if __name__ == "__main__":
    # 1. Load data
    seasonal_data = load_data()

    # 2. EMS params
    ems_params = {
        'DT': 1,
        'cost_load_shed_per_wh': 1,
        'cost_curtailment_per_wh': 0.1,
        'system_eff_pv': 0.95,
        'system_eff_wind': 0.95
    }

    # 3. Run NSGA-II optimization
    results = run_annual_nsga2_optimization(seasonal_data, ems_params)

    # 4. Show results
    print_key_solutions(results, seasonal_data, ems_params)
