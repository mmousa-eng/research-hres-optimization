import numpy as np
import pandas as pd 
from geneticalgorithm import geneticalgorithm as ga
from ems import estimate_operational_performance 
import sys 
import traceback 
import math # For ceil

from models.pv_model import PVModel
from models.wind_model import WindModel
from models.batteries_model import BatteryModel
from models.dg_model import DGModel

# --- Per-Unit Component Ratings ---
PV_MODULE_RATING_W = 1000 
WIND_TURBINE_RATING_W = 20000 
BATTERY_MODULE_ENERGY_WH = 1190 
BATTERY_MODULE_POWER_W = 1190 
DG_UNIT_RATING_W = 2000 

# --- GA Configuration for Continuous Variables ---
DEFAULT_ALGORITHM_PARAMS = {
    'max_num_iteration': 200, 
    'population_size': 50,    
    'mutation_probability': 0.1, 
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform', 
    'max_iteration_without_improv': 20, 
    'funtimeout': 120 
}

BASE_VAR_BOUNDARIES_CONTINUOUS = np.array([
    [2.0, 40.0],    # PV kW
    [0.0, 100.0],   # Wind kW
    [2.0, 59.5],    # Battery kWh
    [5.0, 100.0]    # DG kW
], dtype=float)


DEFAULT_GLOBAL_EMS_PARAMS = {
    'DT': 1,
    'cost_load_shed_per_wh': 5.6e-3, 
    'cost_curtailment_per_wh': 0.1e-3, 
    'system_eff_pv': 0.90,       
    'system_eff_wind': 0.90,     
}
CO2_EMISSION_COST_FACTOR = 0.001 
WEIGHT_INV_COST = 0.5
WEIGHT_OP_COST = 0.2
WEIGHT_EMISSIONS = 0.3
MAX_ALLOWABLE_C_RATE = 1.0 
DG_PEAK_LOAD_SAFETY_FACTOR = 1.02 

MAX_EXPECTED_INV_COST = 200000  
MAX_EXPECTED_ANNUAL_OP_COST = 75000 
MAX_EXPECTED_ANNUAL_CO2 = 30000   


DEBUG_FIRST_FITNESS_CALL = False 
_debug_first_fitness_has_run = False


# Helper function to print detailed objective breakdown
def print_objective_breakdown(
    title, X_capacities_kW, 
    num_pv, num_wind, num_battery, num_dg, 
    dg_capacity_evaluated_kW, derived_battery_power_kW,
    unnorm_inv_cost, unnorm_op_cost, unnorm_co2,
    norm_inv_cost, norm_op_cost, norm_co2,
    final_objective_score
    ):
    print(f"\n--- {title} ---")
    dg_kw_ga_val = X_capacities_kW[3] if len(X_capacities_kW) > 3 else "N/A"
    if isinstance(dg_kw_ga_val, (int, float)):
        dg_kw_ga_str = f"{dg_kw_ga_val:.2f}"
    else:
        dg_kw_ga_str = str(dg_kw_ga_val)

    print(f"  Capacities (GA Output, kW/kWh): PV={X_capacities_kW[0]:.2f}, Wind={X_capacities_kW[1]:.2f}, Batt_kWh={X_capacities_kW[2]:.2f}, DG_kW_GA={dg_kw_ga_str}")
    print(f"  DG Capacity Evaluated: {dg_capacity_evaluated_kW:.2f} kW")
    print(f"  Derived Battery Power: {derived_battery_power_kW:.2f} kW (from {num_battery} modules)")
    print(f"  Number of Units: PV={num_pv}, Wind={num_wind}, Batt={num_battery}, DG={num_dg}")
    print(f"\n  Objective Components Breakdown:")
    w_inv_contrib = WEIGHT_INV_COST * norm_inv_cost
    print(f"  1. Investment Cost:")
    print(f"     - Unnormalized : ${unnorm_inv_cost:,.2f}")
    print(f"     - Max Expected : ${MAX_EXPECTED_INV_COST:,.0f}")
    print(f"     - Normalized   : {norm_inv_cost:.4f}")
    print(f"     - Weight       : {WEIGHT_INV_COST}")
    print(f"     - Contribution : {w_inv_contrib:.4f}")
    w_op_contrib = WEIGHT_OP_COST * norm_op_cost
    print(f"  2. Operational Cost:")
    print(f"     - Unnormalized : ${unnorm_op_cost:,.2f} (Annualized)")
    print(f"     - Max Expected : ${MAX_EXPECTED_ANNUAL_OP_COST:,.0f}")
    print(f"     - Normalized   : {norm_op_cost:.4f}")
    print(f"     - Weight       : {WEIGHT_OP_COST}")
    print(f"     - Contribution : {w_op_contrib:.4f}")
    w_co2_contrib = WEIGHT_EMISSIONS * norm_co2
    print(f"  3. CO2 Emissions:")
    print(f"     - Unnormalized : {unnorm_co2:,.2f} kg (Annualized)")
    print(f"     - Max Expected : {MAX_EXPECTED_ANNUAL_CO2:,.0f} kg")
    print(f"     - Normalized   : {norm_co2:.4f}")
    print(f"     - Weight       : {WEIGHT_EMISSIONS}")
    print(f"     - Contribution : {w_co2_contrib:.4f}")
    
    print(f"\n  Total Weighted Objective Function Score: {final_objective_score:.4f}")
    if all(isinstance(c, (int, float)) for c in [w_inv_contrib, w_op_contrib, w_co2_contrib]):
        print(f"     (Sum of contributions: {w_inv_contrib + w_op_contrib + w_co2_contrib:.4f})\n")
    else:
        print("     (Sum of contributions: Error in calculation - one or more components non-numeric)\n")


def run_annual_ga_optimization(all_seasonal_data, ga_algorithm_params=None, global_ems_params=None):
    global _debug_first_fitness_has_run

    if ga_algorithm_params is None:
        current_ga_params = DEFAULT_ALGORITHM_PARAMS.copy()
    else:
        current_ga_params = ga_algorithm_params.copy()

    if global_ems_params is None:
        current_global_ems_params = DEFAULT_GLOBAL_EMS_PARAMS.copy()
    else:
        current_global_ems_params = global_ems_params.copy()

    max_overall_load_kw = 0
    if all_seasonal_data:
        for season_name_iter, season_df_iter in all_seasonal_data.items():
            if "Load (kW)" in season_df_iter.columns:
                seasonal_max_load = season_df_iter["Load (kW)"].max()
                if pd.notna(seasonal_max_load): 
                    max_overall_load_kw = max(max_overall_load_kw, seasonal_max_load)
    
    if max_overall_load_kw == 0:
        print("WARNING: Max overall load could not be determined or is zero from data. Using a default of 1kW for DG lower bound.")
        max_overall_load_kw = 1.0 

    effective_peak_load_kw_for_dg = max_overall_load_kw * DG_PEAK_LOAD_SAFETY_FACTOR
    
    current_var_boundaries = BASE_VAR_BOUNDARIES_CONTINUOUS.copy()
    current_var_boundaries[3,0] = effective_peak_load_kw_for_dg
    dg_upper_bound_kw = max(effective_peak_load_kw_for_dg * 1.5, effective_peak_load_kw_for_dg + (DG_UNIT_RATING_W * 5 / 1000.0) ) 
    base_dg_upper_bound = BASE_VAR_BOUNDARIES_CONTINUOUS[3,1]
    dg_upper_bound_kw = max(dg_upper_bound_kw, base_dg_upper_bound if base_dg_upper_bound > effective_peak_load_kw_for_dg else effective_peak_load_kw_for_dg * 1.5)
    current_var_boundaries[3,1] = max(dg_upper_bound_kw, effective_peak_load_kw_for_dg + DG_UNIT_RATING_W / 1000.0) 
    
    if current_var_boundaries[3,0] > current_var_boundaries[3,1] : 
        print(f"WARNING: DG lower bound {current_var_boundaries[3,0]:.2f} kW (effective peak) exceeded calculated upper bound. Adjusting upper bound.")
        current_var_boundaries[3,1] = current_var_boundaries[3,0] * 1.2 

    # --- Initialize result dictionary with fallback values ---
    # Capacities from lower bounds of the search space
    fallback_pv_kw = current_var_boundaries[0,0]
    fallback_wind_kw = current_var_boundaries[1,0]
    fallback_batt_kwh = current_var_boundaries[2,0]
    fallback_dg_kw_ga = current_var_boundaries[3,0] # GA would pick at least this
    fallback_dg_kw_eval = max(fallback_dg_kw_ga, effective_peak_load_kw_for_dg)

    fallback_num_pv = math.ceil(fallback_pv_kw * 1000 / PV_MODULE_RATING_W) if PV_MODULE_RATING_W > 0 else 0
    if fallback_pv_kw > 0 and fallback_num_pv == 0: fallback_num_pv = 1
    fallback_num_wind = math.ceil(fallback_wind_kw * 1000 / WIND_TURBINE_RATING_W) if WIND_TURBINE_RATING_W > 0 else 0
    if fallback_wind_kw > 0 and fallback_num_wind == 0: fallback_num_wind = 1
    fallback_num_batt = math.ceil(fallback_batt_kwh * 1000 / BATTERY_MODULE_ENERGY_WH) if BATTERY_MODULE_ENERGY_WH > 0 else 0
    if fallback_batt_kwh > 0 and fallback_num_batt == 0: fallback_num_batt = 1
    fallback_num_dg = math.ceil(fallback_dg_kw_eval * 1000 / DG_UNIT_RATING_W) if DG_UNIT_RATING_W > 0 else 0
    if fallback_dg_kw_eval > 0 and fallback_num_dg == 0: fallback_num_dg = 1
    
    fallback_batt_kw_derived = fallback_num_batt * BATTERY_MODULE_POWER_W / 1000.0

    result_dict = {
        'best_pv_kw': fallback_pv_kw, 
        'best_wind_kw': fallback_wind_kw,
        'best_battery_kwh': fallback_batt_kwh, 
        'best_battery_kw_derived': fallback_batt_kw_derived, 
        'best_dg_kw_ga': fallback_dg_kw_ga, 
        'best_dg_kw_evaluated': fallback_dg_kw_eval, 
        'best_num_pv': fallback_num_pv, 
        'best_num_wind': fallback_num_wind, 
        'best_num_battery': fallback_num_batt, 
        'best_num_dg': fallback_num_dg,
        'objective_value': float('inf'),
        'unnormalized_investment_cost': float('inf'),
        'unnormalized_operational_cost': float('inf'),
        'unnormalized_co2_emissions': float('inf')
    }
    # --- End Initialize result dictionary ---

    def fitness_function(X_capacities):
        effective_peak_load_w_for_dg_local = effective_peak_load_kw_for_dg * 1000
        pv_capacity_kw = X_capacities[0]
        wind_capacity_kw = X_capacities[1]
        battery_energy_kwh = X_capacities[2]
        dg_capacity_kw_ga = X_capacities[3] 

        pv_capacity_w = pv_capacity_kw * 1000
        wind_capacity_w = wind_capacity_kw * 1000
        battery_energy_wh = battery_energy_kwh * 1000
        dg_capacity_w_ga = dg_capacity_kw_ga * 1000
        dg_capacity_w = max(dg_capacity_w_ga, effective_peak_load_w_for_dg_local)

        num_pv = math.ceil(pv_capacity_w / PV_MODULE_RATING_W) if PV_MODULE_RATING_W > 0 else 0
        num_wind = math.ceil(wind_capacity_w / WIND_TURBINE_RATING_W) if WIND_TURBINE_RATING_W > 0 else 0
        num_battery = math.ceil(battery_energy_wh / BATTERY_MODULE_ENERGY_WH) if BATTERY_MODULE_ENERGY_WH > 0 else 0
        num_dg = math.ceil(dg_capacity_w / DG_UNIT_RATING_W) if DG_UNIT_RATING_W > 0 else 0
        
        if pv_capacity_w > 0 and num_pv == 0: num_pv = 1
        if wind_capacity_w > 0 and num_wind == 0: num_wind = 1
        if battery_energy_wh > 0 and num_battery == 0: num_battery = 1
        if dg_capacity_w > 0 and num_dg == 0: num_dg = 1 
        if battery_energy_wh > 0 and battery_energy_wh < BATTERY_MODULE_ENERGY_WH / 2 and num_battery == 0 :
             num_battery = 1

        dummy_series = np.array([0]) 
        pv_model_for_cost = PVModel(num_modules=num_pv, G=dummy_series, Ta=dummy_series, V=dummy_series)
        wind_model_for_cost = WindModel(num_turbines=num_wind, V=dummy_series)
        battery_model_for_cost = BatteryModel(num_batteries=num_battery, DT=1, num_steps=1) 
        dg_model_for_cost = DGModel(num_generators=num_dg, DT=1, num_steps=1) 

        inv_cost_pv = pv_model_for_cost.compute_total_cost(num_pv)
        inv_cost_wind = wind_model_for_cost.compute_total_cost(num_wind)
        inv_cost_battery = battery_model_for_cost.compute_total_cost()
        inv_cost_dg = dg_model_for_cost.compute_total_cost()
        total_annualized_investment_cost = inv_cost_pv + inv_cost_wind + inv_cost_battery + inv_cost_dg

        total_annual_op_cost = 0
        total_annual_co2 = 0
        num_simulated_weeks = 0

        for season_name_iter, season_df_iter in all_seasonal_data.items():
            num_hours_iter = len(season_df_iter)
            g_hourly_iter = season_df_iter["G(W/m^2)"].values
            ta_hourly_iter = season_df_iter["Ta(C) at 2m"].values
            v_hourly_iter = season_df_iter["V(m/s) at 10m"].values

            pv_sim_model = PVModel(num_modules=num_pv, G=g_hourly_iter, Ta=ta_hourly_iter, V=v_hourly_iter)
            wind_sim_model = WindModel(num_turbines=num_wind, V=v_hourly_iter)
            battery_sim_model = BatteryModel(num_batteries=num_battery, DT=current_global_ems_params['DT'], num_steps=num_hours_iter)
            dg_sim_model = DGModel(num_generators=num_dg, DT=current_global_ems_params['DT'], num_steps=num_hours_iter)

            pv_power_w_hourly_iter = pv_sim_model.power_output() 
            wind_power_w_hourly_iter = wind_sim_model.power_output()

            op_results = estimate_operational_performance(
                season_df=season_df_iter,
                pv_power_available_w=pv_power_w_hourly_iter, 
                wind_power_available_w=wind_power_w_hourly_iter, 
                battery_model=battery_sim_model, 
                dg_model=dg_sim_model,          
                global_params=current_global_ems_params
            )
            
            if not all_seasonal_data or len(all_seasonal_data) == 0: 
                scaling_factor_op_cost = 52 
            else:
                scaling_factor_op_cost = (52 / len(all_seasonal_data))

            total_annual_op_cost += op_results['weekly_op_cost'] * scaling_factor_op_cost
            total_annual_co2 += op_results['weekly_co2'] * scaling_factor_op_cost
            num_simulated_weeks +=1
            
        if num_simulated_weeks == 0 : 
            return float('inf')

        norm_inv_cost = max(0, total_annualized_investment_cost) / MAX_EXPECTED_INV_COST if MAX_EXPECTED_INV_COST > 0 else 0
        norm_op_cost = max(0, total_annual_op_cost) / MAX_EXPECTED_ANNUAL_OP_COST if MAX_EXPECTED_ANNUAL_OP_COST > 0 else 0
        norm_co2 = max(0, total_annual_co2) / MAX_EXPECTED_ANNUAL_CO2 if MAX_EXPECTED_ANNUAL_CO2 > 0 else 0
        
        norm_inv_cost = min(norm_inv_cost, 1.5) 
        norm_op_cost = min(norm_op_cost, 1.5)
        norm_co2 = min(norm_co2, 1.5)

        objective = (WEIGHT_INV_COST * norm_inv_cost +
                     WEIGHT_OP_COST * norm_op_cost +
                     WEIGHT_EMISSIONS * norm_co2)
        
        global _debug_first_fitness_has_run 
        if DEBUG_FIRST_FITNESS_CALL and not _debug_first_fitness_has_run:
            derived_battery_power_kw = num_battery * BATTERY_MODULE_POWER_W / 1000.0
            print_objective_breakdown(
                title="DEBUG Fitness Call (First Evaluation)",
                X_capacities_kW=X_capacities, 
                num_pv=num_pv, num_wind=num_wind, num_battery=num_battery, num_dg=num_dg,
                dg_capacity_evaluated_kW=dg_capacity_w / 1000.0, 
                derived_battery_power_kW=derived_battery_power_kw,
                unnorm_inv_cost=total_annualized_investment_cost,
                unnorm_op_cost=total_annual_op_cost,
                unnorm_co2=total_annual_co2,
                norm_inv_cost=norm_inv_cost, norm_op_cost=norm_op_cost, norm_co2=norm_co2,
                final_objective_score=objective
            )
            _debug_first_fitness_has_run = True
        return objective

    if DEBUG_FIRST_FITNESS_CALL and not _debug_first_fitness_has_run :
        print(f"DEBUG: Original Max overall load determined: {max_overall_load_kw:.2f} kW")
        print(f"DEBUG: Effective Peak Load for DG (with safety factor): {effective_peak_load_kw_for_dg:.2f} kW")
        print(f"DEBUG: DG capacity bounds for GA (kW): Min={current_var_boundaries[3,0]:.2f}, Max={current_var_boundaries[3,1]:.2f}")
        print(f"DEBUG: Preparing to manually test fitness_function with initial capacities...")
        fixed_X_capacities = [current_var_boundaries[i][0] for i in range(current_var_boundaries.shape[0])] 
        print(f"DEBUG: Testing with GA capacities (lower bounds, kW/kWh): {fixed_X_capacities}")
        try:
            print("DEBUG: Calling fitness_function manually...")
            score = fitness_function(fixed_X_capacities) 
            print(f"DEBUG: Manual fitness_function score: {score}")
        except Exception as e:
            print(f"DEBUG: Exception during manual fitness_function test: {e}")
            traceback.print_exc()
        raise SystemExit(f"Exiting after manual debug of first fitness call.")

    print(f"DEBUG: GA PARAMS BEING USED for Annual Opt: {current_ga_params}")
    print(f"DEBUG: Original Max Overall Load: {max_overall_load_kw:.2f} kW")
    print(f"DEBUG: Effective Peak Load for DG (with safety factor {DG_PEAK_LOAD_SAFETY_FACTOR}): {effective_peak_load_kw_for_dg:.2f} kW")
    print(f"DEBUG: Final DG capacity bounds for GA (kW): Min={current_var_boundaries[3,0]:.2f}, Max={current_var_boundaries[3,1]:.2f}")

    model = ga(
        function=fitness_function,
        dimension=4, 
        variable_type='real', 
        variable_boundaries=current_var_boundaries, 
        algorithm_parameters=current_ga_params
    )

    print(f"Running Annual GA (Iterations: {current_ga_params['max_num_iteration']}, Pop_size: {current_ga_params['population_size']})...")
    
    try:
        model.run()
        if model.output_dict and 'variable' in model.output_dict and 'function' in model.output_dict:
            best_capacities_result_kw = model.output_dict['variable'] 
            result_dict['objective_value'] = model.output_dict['function']   
            
            result_dict['best_pv_kw'] = best_capacities_result_kw[0]
            result_dict['best_wind_kw'] = best_capacities_result_kw[1]
            result_dict['best_battery_kwh'] = best_capacities_result_kw[2]
            result_dict['best_dg_kw_ga'] = best_capacities_result_kw[3]
            result_dict['best_dg_kw_evaluated'] = max(result_dict['best_dg_kw_ga'], effective_peak_load_kw_for_dg)
            
            # Recalculate nums and derived power for the final result_dict
            num_pv_opt = math.ceil(result_dict['best_pv_kw'] * 1000 / PV_MODULE_RATING_W) if PV_MODULE_RATING_W > 0 else 0
            if result_dict['best_pv_kw'] > 0 and num_pv_opt == 0: num_pv_opt = 1
            result_dict['best_num_pv'] = num_pv_opt

            num_wind_opt = math.ceil(result_dict['best_wind_kw'] * 1000 / WIND_TURBINE_RATING_W) if WIND_TURBINE_RATING_W > 0 else 0
            if result_dict['best_wind_kw'] > 0 and num_wind_opt == 0: num_wind_opt = 1
            result_dict['best_num_wind'] = num_wind_opt
            
            num_batt_opt = math.ceil(result_dict['best_battery_kwh'] * 1000 / BATTERY_MODULE_ENERGY_WH) if BATTERY_MODULE_ENERGY_WH > 0 else 0
            if result_dict['best_battery_kwh'] > 0 and num_batt_opt == 0: num_batt_opt = 1
            if result_dict['best_battery_kwh'] > 0 and result_dict['best_battery_kwh'] * 1000 < BATTERY_MODULE_ENERGY_WH / 2 and num_batt_opt == 0 : num_batt_opt = 1
            result_dict['best_num_battery'] = num_batt_opt
            result_dict['best_battery_kw_derived'] = num_batt_opt * BATTERY_MODULE_POWER_W / 1000.0

            num_dg_opt = math.ceil(result_dict['best_dg_kw_evaluated'] * 1000 / DG_UNIT_RATING_W) if DG_UNIT_RATING_W > 0 else 0
            if result_dict['best_dg_kw_evaluated'] > 0 and num_dg_opt == 0: num_dg_opt = 1
            result_dict['best_num_dg'] = num_dg_opt

            print("GA run completed for Annual Optimization.")
            # --- Re-calculate all cost components for the best solution to print and return ---
            dummy_series = np.array([0])
            pv_mfc = PVModel(num_modules=result_dict['best_num_pv'], G=dummy_series, Ta=dummy_series, V=dummy_series)
            wind_mfc = WindModel(num_turbines=result_dict['best_num_wind'], V=dummy_series)
            batt_mfc = BatteryModel(num_batteries=result_dict['best_num_battery'], DT=1, num_steps=1)
            dg_mfc = DGModel(num_generators=result_dict['best_num_dg'], DT=1, num_steps=1)
            
            result_dict['unnormalized_investment_cost'] = pv_mfc.compute_total_cost(result_dict['best_num_pv']) + \
                                                          wind_mfc.compute_total_cost(result_dict['best_num_wind']) + \
                                                          batt_mfc.compute_total_cost() + \
                                                          dg_mfc.compute_total_annualized_cost()
            
            annual_op = 0
            annual_cO2 = 0
            sim_weeks = 0
            for s_name, s_df_iter in all_seasonal_data.items():
                s_g = s_df_iter["G(W/m^2)"].values; s_ta = s_df_iter["Ta(C) at 2m"].values; s_v = s_df_iter["V(m/s) at 10m"].values
                s_pv_model = PVModel(num_modules=result_dict['best_num_pv'], G=s_g, Ta=s_ta, V=s_v)
                s_wind_model = WindModel(num_turbines=result_dict['best_num_wind'], V=s_v)
                s_batt_model = BatteryModel(num_batteries=result_dict['best_num_battery'], DT=current_global_ems_params['DT'], num_steps=len(s_df_iter))
                s_dg_model = DGModel(num_generators=result_dict['best_num_dg'], DT=current_global_ems_params['DT'], num_steps=len(s_df_iter))
                s_pv_power = s_pv_model.power_output()
                s_wind_power = s_wind_model.power_output()
                s_op_res = estimate_operational_performance(s_df_iter, s_pv_power, s_wind_power, s_batt_model, s_dg_model, current_global_ems_params)
                op_scale = 52 / len(all_seasonal_data) if all_seasonal_data else 52
                annual_op += s_op_res['weekly_op_cost'] * op_scale
                annual_cO2 += s_op_res['weekly_co2'] * op_scale
                sim_weeks +=1
            
            result_dict['unnormalized_operational_cost'] = annual_op if sim_weeks > 0 else float('inf')
            result_dict['unnormalized_co2_emissions'] = annual_cO2 if sim_weeks > 0 else float('inf')

            norm_i = max(0, result_dict['unnormalized_investment_cost']) / MAX_EXPECTED_INV_COST if MAX_EXPECTED_INV_COST > 0 else 0
            norm_o = max(0, result_dict['unnormalized_operational_cost']) / MAX_EXPECTED_ANNUAL_OP_COST if MAX_EXPECTED_ANNUAL_OP_COST > 0 else 0
            norm_c = max(0, result_dict['unnormalized_co2_emissions']) / MAX_EXPECTED_ANNUAL_CO2 if MAX_EXPECTED_ANNUAL_CO2 > 0 else 0
            norm_i, norm_o, norm_c = min(norm_i, 1.5), min(norm_o, 1.5), min(norm_c, 1.5)
            
            print_objective_breakdown(
                title="Detailed Breakdown for Best Solution Found by GA",
                X_capacities_kW=best_capacities_result_kw,
                num_pv=result_dict['best_num_pv'], num_wind=result_dict['best_num_wind'], 
                num_battery=result_dict['best_num_battery'], num_dg=result_dict['best_num_dg'],
                dg_capacity_evaluated_kW=result_dict['best_dg_kw_evaluated'],
                derived_battery_power_kW=result_dict['best_battery_kw_derived'],
                unnorm_inv_cost=result_dict['unnormalized_investment_cost'], 
                unnorm_op_cost=result_dict['unnormalized_operational_cost'], 
                unnorm_co2=result_dict['unnormalized_co2_emissions'],
                norm_inv_cost=norm_i, norm_op_cost=norm_o, norm_co2=norm_c,
                final_objective_score=result_dict['objective_value']
            )
        else:
            print("GA run did not produce a valid output_dict. Returning fallback solution.")
            # Fallback values are already in result_dict from initialization

    except Exception as e:
        print(f"ERROR: Exception during GA model.run(): {e}")
        traceback.print_exc()
        print(f"GA for Annual Opt failed. Returning fallback solution.")
        # Fallback values are already in result_dict from initialization
        
    return result_dict
