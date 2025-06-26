import numpy as np
import math

def estimate_operational_performance(
    season_df,
    pv_power_available_w,
    wind_power_available_w,
    battery_model,
    dg_model,
    global_params
):
    """
    Estimates weekly operational performance using a simplified greedy dispatch logic
    with flexible operation of diesel generator units.
    """

    dg_hours_active = 0
    dg_energy_total_wh = 0
    total_load_wh = 0
    num_hours = len(season_df)
    dt = global_params['DT']
    load_demand_w = season_df["Load (kW)"].values * 1000

    pv_power_dispatchable_w = pv_power_available_w * global_params.get('system_eff_pv', 1.0) * 0.98
    wind_power_dispatchable_w = wind_power_available_w * global_params.get('system_eff_wind', 1.0) * 0.98

    total_dg_fuel_cost_usd = 0
    total_dg_om_cost_usd = 0
    total_battery_degradation_cost = 0
    total_load_shedding_cost = 0
    total_curtailment_cost = 0
    total_co2_emissions_kg = 0
    
    batt_cap_cost_per_batt = battery_model.capital_cost
    batt_life_cycles = battery_model.lifetime_cycles
    batt_energy_cap_wh_per_batt = battery_model.energy_capacity
    if batt_life_cycles > 0 and batt_energy_cap_wh_per_batt > 0:
        degradation_cost_per_wh_throughput = batt_cap_cost_per_batt / (batt_life_cycles * batt_energy_cap_wh_per_batt)
    else:
        degradation_cost_per_wh_throughput = float('inf') 

    for t in range(num_hours):
        load_w_t = load_demand_w[t]
        total_load_wh += load_w_t * dt

        pv_w_t = pv_power_dispatchable_w[t]
        wind_w_t = wind_power_dispatchable_w[t]

        remaining_load_w = load_w_t
        pv_used_w = 0
        wind_used_w = 0

        use_pv = min(remaining_load_w, pv_w_t)
        remaining_load_w -= use_pv
        pv_used_w += use_pv
        
        use_wind = min(remaining_load_w, wind_w_t)
        remaining_load_w -= use_wind
        wind_used_w += use_wind

        surplus_pv_w = pv_w_t - pv_used_w
        surplus_wind_w = wind_w_t - wind_used_w
        total_surplus_re_w = surplus_pv_w + surplus_wind_w

        batt_charge_w = 0
        batt_discharge_w = 0
        
        # Battery charging with constraint on max charge power and converter efficiency
        if total_surplus_re_w > 0 and battery_model.current_soc < battery_model.soc_max:
            max_charge_power_w = battery_model.num_batteries * battery_model.max_charge / battery_model.converter_efficiency
            energy_to_soc_max_wh = (battery_model.soc_max - battery_model.current_soc) * battery_model.num_batteries * battery_model.energy_capacity
            max_charge_power_soc_limit_w = energy_to_soc_max_wh / (dt * battery_model.converter_efficiency) if dt > 0 and battery_model.converter_efficiency > 0 else float('inf')
            actual_charge_w = min(total_surplus_re_w, max_charge_power_w, max_charge_power_soc_limit_w)
            battery_model.charge_battery(actual_charge_w, dt)
            batt_charge_w = actual_charge_w
            
            charged_from_pv = min(actual_charge_w, surplus_pv_w)
            pv_used_w += charged_from_pv
            remaining_charge_needed = actual_charge_w - charged_from_pv
            if remaining_charge_needed > 0:
                charged_from_wind = min(remaining_charge_needed, surplus_wind_w)
                wind_used_w += charged_from_wind

        # Battery discharging with constraint on max discharge power and converter efficiency
        if remaining_load_w > 0 and battery_model.current_soc > battery_model.soc_min:
            max_discharge_power_w = battery_model.num_batteries * battery_model.max_discharge * battery_model.converter_efficiency
            energy_from_soc_min_wh = (battery_model.current_soc - battery_model.soc_min) * battery_model.num_batteries * battery_model.energy_capacity
            max_discharge_power_soc_limit_w = (energy_from_soc_min_wh * battery_model.converter_efficiency) / dt if dt > 0 else float('inf')
            actual_discharge_w = min(remaining_load_w, max_discharge_power_w, max_discharge_power_soc_limit_w)
            battery_model.discharge_battery(actual_discharge_w, dt)
            batt_discharge_w = actual_discharge_w
            remaining_load_w -= actual_discharge_w

        energy_charged_cells_wh = batt_charge_w * battery_model.converter_efficiency * dt
        energy_discharged_cells_wh = (batt_discharge_w / battery_model.converter_efficiency) * dt if battery_model.converter_efficiency > 0 else float('inf')
        total_batt_throughput_wh = energy_charged_cells_wh + energy_discharged_cells_wh
        total_battery_degradation_cost += total_batt_throughput_wh * degradation_cost_per_wh_throughput

        dg_power_delivered_w_t = 0
        num_dg_active_this_hour = 0

        if remaining_load_w > 0 and dg_model.num_generators > 0:
            dg_dispatch_needed_w = remaining_load_w
            max_total_dg_capacity_w = dg_model.num_generators * dg_model.rated_power
            dg_power_delivered_w_t = min(dg_dispatch_needed_w, max_total_dg_capacity_w)

            if dg_power_delivered_w_t > 0:
                dg_hours_active += 1
                dg_energy_total_wh += dg_power_delivered_w_t * dt

                if dg_model.rated_power == 0:
                    num_dg_active_this_hour = dg_model.num_generators 
                else:
                    num_dg_active_this_hour = math.ceil(dg_power_delivered_w_t / dg_model.rated_power)

                num_dg_active_this_hour = min(num_dg_active_this_hour, dg_model.num_generators)

        base_fuel_L = 0
        if num_dg_active_this_hour > 0:
            base_fuel_L = num_dg_active_this_hour * (dg_model.base_load_consumption * dg_model.rated_power * dt)

        variable_fuel_L = dg_model.fuel_consumption_factor * dg_power_delivered_w_t * dt
        current_hour_fuel_L = variable_fuel_L + base_fuel_L

        current_hour_fuel_price_cost = current_hour_fuel_L * dg_model.fuel_price_per_liter
        current_hour_co2_kg = current_hour_fuel_L * dg_model.co2_emission_factor * dg_model.diesel_cfactor

        total_dg_fuel_cost_usd += current_hour_fuel_price_cost
        total_co2_emissions_kg += current_hour_co2_kg

        remaining_load_w -= dg_power_delivered_w_t
            
        load_shed_w_t = max(0, remaining_load_w)
        if load_shed_w_t > 0:
            total_load_shedding_cost += load_shed_w_t * global_params['cost_load_shed_per_wh'] * dt

        curtailed_pv_w = pv_power_dispatchable_w[t] - pv_used_w
        curtailed_wind_w = wind_power_dispatchable_w[t] - wind_used_w
        total_curtailment_w_t = max(0, curtailed_pv_w) + max(0, curtailed_wind_w)
        if total_curtailment_w_t > 0:
            total_curtailment_cost += total_curtailment_w_t * global_params['cost_curtailment_per_wh'] * dt
            
    total_weekly_op_cost = (
        total_dg_fuel_cost_usd +
        total_dg_om_cost_usd +
        total_battery_degradation_cost +
        total_load_shedding_cost +
        total_curtailment_cost
    )
    
    # Return also DG total energy and total load energy for contribution calculation
    return {
        'weekly_op_cost': total_weekly_op_cost,
        'weekly_co2': total_co2_emissions_kg,
        'total_dg_energy_wh': dg_energy_total_wh,
        'total_load_energy_wh': total_load_wh
    }
