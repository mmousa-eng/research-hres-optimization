import numpy as np
from models.pv_model import PVModel
from models.wind_model import WMTodel
from models.batteries_model import BatteryModel
from models.dg_model import DGModel

def ems_dispatch(pv_power, wind_power, load_demand, 
                 battery_model, dg_model, DT, eff_inv=0.95):
    from models.batteries_model import BatteryModel
    from models.dg_model import DGModel

def ems_dispatch(pv_power, wind_power, load_demand, 
                battery_model: BatteryModel, 
                dg_model: DGModel, 
                DT, eff_inv=0.95):

    """
    EMS dispatch algorithm for each hour.

    Prioritizes renewable energy. Solves MILP for each hour.

    Returns:
        dict: Results per time step (dicts of lists)
    """
    num_steps = len(load_demand)

    # Store results over the time horizon
    results = {
        'battery_charge': [],
        'battery_discharge': [],
        'dg_output': [],
        'power_sh': [],
        'power_cur': [],
        'soc': [],
        'operating_cost_total': [],
        'co2_emissions': []
    }

    for t in range(num_steps):
        # Time step inputs
        pv = pv_power[t] * battery_model.converter_efficiency
        wind = wind_power[t] * battery_model.converter_efficiency
        load = load_demand[t] / eff_inv
        soc = battery_model.current_soc

        # Available renewable energy
        renewable_gen = pv + wind

        # System losses (2% of renewable)
        losses = 0.02 * renewable_gen

        # Initialize dispatch variables (basic greedy prioritization)
        power_sh = 0  # Load shedding
        power_cur = 0  # Curtailment
        charge = 0
        discharge = 0
        dg_output = 0

        # Use renewable first
        supply = min(renewable_gen, load)
        residual = load - supply

        # Try to discharge battery to cover residual
        max_discharge_power = battery_model.max_discharge * battery_model.num_batteries
        energy_available = (soc - battery_model.soc_min) * battery_model.num_batteries * battery_model.energy_capacity
        max_discharge_energy = min(energy_available, max_discharge_power * DT)
        discharge = min(residual, max_discharge_energy / DT)
        battery_model.discharge_battery(discharge, DT)
        residual -= discharge

        # Use DG if residual still exists
        if residual > 0:
            dg_limit = dg_model.num_generators * dg_model.efficiency * dg_model.rated_power
            dg_output = min(residual, dg_limit)
            residual -= dg_output

        # Remaining residual = load shedding
        power_sh = residual if residual > 0 else 0

        # If excess renewable energy remains, try to charge battery
        excess_renew = max(0, renewable_gen - load)
        max_charge_power = battery_model.max_charge * battery_model.num_batteries
        energy_needed = (battery_model.soc_max - battery_model.current_soc) * battery_model.num_batteries * battery_model.energy_capacity
        max_charge_energy = min(energy_needed, max_charge_power * DT)
        charge = min(excess_renew, max_charge_energy / DT)
        battery_model.charge_battery(charge, DT)

        # Remaining unused renewable = curtailment
        power_cur = max(0, excess_renew - charge)

        # Store results
        dg_results = dg_model.power_output(dg_output)
        cost_sh = 5.6e-3 * power_sh * DT
        cost_cur = 5.6e-3 * power_cur * DT

        results['battery_charge'].append(charge)
        results['battery_discharge'].append(discharge)
        results['dg_output'].append(dg_output)
        results['power_sh'].append(power_sh)
        results['power_cur'].append(power_cur)
        results['soc'].append(battery_model.current_soc)
        results['operating_cost_total'].append(dg_results['operating_cost_usd'] + cost_sh + cost_cur)
        results['co2_emissions'].append(dg_results['co2_emissions_kg'])

    return results

def run_ems(season_df, pv_size, wind_size, battery_size, dg_size):
    print(f"[DEBUG] Running EMS for: PV={pv_size}, Wind={wind_size}, Batt={battery_size}, DG={dg_size}")
    DT = 1  # hour
    num_steps = len(season_df)

    # === Inputs from data ===
    G = season_df["G(W/m^2)"]
    Ta = season_df["Ta(C) at 2m"]
    V = season_df["V(m/s) at 10m"]
    load_kw = season_df["Load (kW)"]
    load_w = load_kw * 1000

    # === Instantiate models ===
    pv_model = PVModel(num_modules=pv_size, G=G, Ta=Ta, V=V)
    wind_model = WMTodel(num_turbines=wind_size, V=V)
    battery_model = BatteryModel(num_batteries=battery_size, DT=DT, num_steps=num_steps)
    dg_model = DGModel(num_generators=dg_size, DT=DT, num_steps=num_steps)

    # === Efficiency Parameters ===
    inverter_eff = 0.95
    pv_eff = 0.8
    wind_eff = 0.85
    Cls = 5.6e-3  # $/Wh
    Crc = 5.6e-3  # $/Wh

    # === Time Series Results Storage ===
    total_ls_cost = 0
    total_cur_cost = 0
    total_dg_fuel_cost = 0
    total_dg_emissions = 0

    pv_power = pv_model.power_output()
    wind_power = wind_model.power_output()
    net_renewable = (pv_power * pv_eff + wind_power * wind_eff) * (1 - 0.02)  # after losses

    for t in range(num_steps):
        load = load_w.iloc[t]
        renewable_avail = net_renewable.iloc[t]

        # Prioritize renewable
        used_renewable = min(load, renewable_avail)
        remaining_load = load - used_renewable

        # Dispatch battery
        battery_power = 0
        if remaining_load > 0:
            battery_power = min(battery_model.max_discharge * battery_size, remaining_load)
            battery_model.discharge_battery(battery_power, DT)
        else:
            charge_power = min(battery_model.max_charge * battery_size, -remaining_load)
            battery_model.charge_battery(charge_power, DT)

        remaining_load -= battery_power

        # Dispatch DG if needed
        dg_dispatch = 0
        if remaining_load > 0:
            dg_dispatch = min(remaining_load, dg_size * dg_model.efficiency * dg_model.rated_power * 1000)
            result = dg_model.power_output(dg_dispatch / 1000)  # convert W to kW
            total_dg_fuel_cost += result['operating_cost_usd']
            total_dg_emissions += result['co2_emissions_kg']
            remaining_load -= dg_dispatch

        # Load Shedding if still unmet
        power_sh = max(0, remaining_load)
        cost_ls = Cls * power_sh * DT
        total_ls_cost += cost_ls

        # Curtailment if renewable not used fully
        power_cur = max(0, renewable_avail - used_renewable)
        cost_cur = Crc * power_cur * DT
        total_cur_cost += cost_cur

    total_operating_cost = total_ls_cost + total_cur_cost + total_dg_fuel_cost
    print(f"[DEBUG] Returning from EMS: PV={pv_size}, Wind={wind_size}, Batt={battery_size}, DG={dg_size}")
    scaling_factor = 50
    return {
        'total_operating_cost': scaling_factor*total_operating_cost,
        'load_shedding_cost': scaling_factor*total_ls_cost,
        'curtailment_cost': scaling_factor*total_cur_cost,
        'total_dg_fuel_cost': scaling_factor*total_dg_fuel_cost,
        'total_dg_emissions': scaling_factor*total_dg_emissions
    }
