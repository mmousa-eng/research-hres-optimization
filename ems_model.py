import numpy as np

def ems_optimizer(nb, ndg, npv, nwt, df, battery_capacity, dg_capacity, eta_charge, eta_discharge, delta_t, weight_cost, weight_emissions):
    """
    Energy Management System (EMS) for optimized power dispatch.

    Args:
        nb (int): Number of batteries
        ndg (int): Number of diesel generators
        npv (int): Number of PV modules
        nwt (int): Number of wind turbines
        df (pandas.DataFrame): Seasonal data (load, renewable generation).
        battery_capacity (float): Total battery energy storage capacity.
        dg_capacity (float): Diesel generator capacity.
        eta_charge (float): Battery charging efficiency.
        eta_discharge (float): Battery discharging efficiency.
        delta_t (float): Time interval.
        weight_cost (float): Weight for cost minimization.
        weight_emissions (float): Weight for emissions reduction.

    Returns:
        dict: Optimized energy dispatch per timestep.
    """
    
    optimized_dispatch = []
    battery_soc = battery_capacity * 0.8  # Assume 80% initial SoC (80% capacity)

    for k in range(len(df)):
        load = df["Load (kW)"].iloc[k]
        pv_gen = df["G(W/m^2)"].iloc[k] * npv
        wind_gen = df["V(m/s) at 10m"].iloc[k] * nwt
        
        # **Step 1: Prioritize renewables**
        total_renewable = pv_gen + wind_gen
        remaining_load = max(0, load - total_renewable)

        # **Step 2: Optimize battery use**
        battery_discharge = min(remaining_load, battery_soc * eta_discharge)
        battery_soc -= battery_discharge

        # **Step 3: Use diesel only if necessary**
        diesel_power = max(0, remaining_load - battery_discharge)

        # **Step 4: Battery SoC update**
        renewable_surplus = max(0, pv_gen + wind_gen - load)
        battery_charge = min(battery_capacity - battery_soc, renewable_surplus * eta_charge)
        battery_soc += battery_charge

        # **Step 5: Cost & Emissions (weighted factors)**
        cost_diesel = diesel_power * weight_cost  # Simplified cost function
        emissions_diesel = diesel_power * weight_emissions  # CO₂ emissions weighting

        optimized_dispatch.append({
            "Hour": k,
            "Load (kW)": load,
            "PV Power (kWh)": pv_gen,
            "Wind Power (kWh)": wind_gen,
            "Battery Discharge (kWh)": battery_discharge,
            "Diesel Generator Power (kWh)": diesel_power,
            "Battery SoC": battery_soc,
            "Diesel Cost": cost_diesel,
            "Diesel Emissions": emissions_diesel
        })

    return optimized_dispatch