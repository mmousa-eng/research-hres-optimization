import numpy as np
import math
from scipy.optimize import linprog


def estimate_operational_performance(
    season_df,
    pv_power_available_w,
    wind_power_available_w,
    battery_model,
    dg_model,
    global_params,
    load_demand_w=None
):
    """
    Estimates weekly operational performance using a rule-based greedy dispatch logic
    (PV -> Wind -> Battery -> DG -> Load Shedding) with flexible operation of modular
    diesel generator units.
    """
    dg_hours_active = 0
    dg_energy_total_wh = 0.0
    total_load_wh = 0.0
    dt = global_params['DT']
    if load_demand_w is None:
        load_demand_w = season_df["Load (kW)"].values * 1000.0
    num_hours = len(load_demand_w)

    pv_power_dispatchable_w = pv_power_available_w * global_params.get('system_eff_pv', 1.0) * 0.98
    wind_power_dispatchable_w = wind_power_available_w * global_params.get('system_eff_wind', 1.0) * 0.98

    total_dg_fuel_cost_usd = 0.0
    total_dg_om_cost_usd = 0.0
    total_battery_degradation_cost = 0.0
    total_load_shedding_cost = 0.0
    total_curtailment_cost = 0.0
    total_co2_emissions_kg = 0.0

    batt_cap_cost_per_batt = battery_model.capital_cost
    batt_life_cycles = battery_model.lifetime_cycles
    batt_energy_cap_wh_per_batt = battery_model.energy_capacity
    if batt_life_cycles > 0 and batt_energy_cap_wh_per_batt > 0:
        degradation_cost_per_wh_throughput = batt_cap_cost_per_batt / (
            batt_life_cycles * batt_energy_cap_wh_per_batt
        )
    else:
        degradation_cost_per_wh_throughput = float('inf')

    num_batt = battery_model.num_batteries
    batt_total_cap_wh = num_batt * batt_energy_cap_wh_per_batt
    conv_eff = battery_model.converter_efficiency
    soc_min = battery_model.soc_min
    soc_max = battery_model.soc_max
    current_soc = battery_model.current_soc
    max_charge_power_w = (num_batt * battery_model.max_charge / conv_eff) if conv_eff > 0 else float('inf')
    max_discharge_power_w = num_batt * battery_model.max_discharge * conv_eff

    num_dg = dg_model.num_generators
    dg_rated_w = dg_model.rated_power
    max_total_dg_capacity_w = num_dg * dg_rated_w
    base_load_cons = dg_model.base_load_consumption
    fuel_cons_factor = dg_model.fuel_consumption_factor
    fuel_price = dg_model.fuel_price_per_liter
    co2_factor = dg_model.co2_emission_factor * dg_model.diesel_cfactor
    cost_load_shed = global_params['cost_load_shed_per_wh']
    cost_curt = global_params['cost_curtailment_per_wh']

    for t in range(num_hours):
        load_w_t = float(load_demand_w[t])
        total_load_wh += load_w_t * dt

        pv_w_t = float(pv_power_dispatchable_w[t])
        wind_w_t = float(wind_power_dispatchable_w[t])

        remaining_load_w = load_w_t
        use_pv = remaining_load_w if remaining_load_w < pv_w_t else pv_w_t
        remaining_load_w -= use_pv
        pv_used_w = use_pv

        use_wind = remaining_load_w if remaining_load_w < wind_w_t else wind_w_t
        remaining_load_w -= use_wind
        wind_used_w = use_wind

        surplus_pv_w = pv_w_t - pv_used_w
        surplus_wind_w = wind_w_t - wind_used_w
        total_surplus_re_w = surplus_pv_w + surplus_wind_w

        batt_charge_w = 0.0
        batt_discharge_w = 0.0

        # Battery charging with constraint on max charge power and converter efficiency (Eq. 29)
        if total_surplus_re_w > 0.0 and current_soc < soc_max and batt_total_cap_wh > 0:
            energy_to_soc_max_wh = (soc_max - current_soc) * batt_total_cap_wh
            max_charge_power_soc_limit_w = (
                energy_to_soc_max_wh / (dt * conv_eff)
                if dt > 0 and conv_eff > 0 else float('inf')
            )
            actual_charge_w = min(total_surplus_re_w, max_charge_power_w, max_charge_power_soc_limit_w)
            delta_soc = (actual_charge_w * dt * conv_eff) / batt_total_cap_wh
            current_soc = min(current_soc + delta_soc, soc_max)
            batt_charge_w = actual_charge_w

            charged_from_pv = min(actual_charge_w, surplus_pv_w)
            pv_used_w += charged_from_pv
            remaining_charge_needed = actual_charge_w - charged_from_pv
            if remaining_charge_needed > 0.0:
                charged_from_wind = min(remaining_charge_needed, surplus_wind_w)
                wind_used_w += charged_from_wind

        # Battery discharging with constraint on max discharge power and converter efficiency (Eq. 29)
        if remaining_load_w > 0.0 and current_soc > soc_min and batt_total_cap_wh > 0:
            energy_from_soc_min_wh = (current_soc - soc_min) * batt_total_cap_wh
            max_discharge_power_soc_limit_w = (energy_from_soc_min_wh * conv_eff) / dt if dt > 0 else float('inf')
            actual_discharge_w = min(remaining_load_w, max_discharge_power_w, max_discharge_power_soc_limit_w)
            delta_soc = (actual_discharge_w * dt / conv_eff) / batt_total_cap_wh
            current_soc = max(current_soc - delta_soc, soc_min)
            batt_discharge_w = actual_discharge_w
            remaining_load_w -= actual_discharge_w

        energy_charged_cells_wh = batt_charge_w * conv_eff * dt
        energy_discharged_cells_wh = (batt_discharge_w / conv_eff) * dt if conv_eff > 0 else float('inf')
        total_batt_throughput_wh = energy_charged_cells_wh + energy_discharged_cells_wh
        total_battery_degradation_cost += total_batt_throughput_wh * degradation_cost_per_wh_throughput

        dg_power_delivered_w_t = 0.0
        num_dg_active_this_hour = 0

        if remaining_load_w > 0.0 and num_dg > 0:
            dg_dispatch_needed_w = remaining_load_w
            dg_power_delivered_w_t = min(dg_dispatch_needed_w, max_total_dg_capacity_w)

            if dg_power_delivered_w_t > 0.0:
                dg_hours_active += 1
                dg_energy_total_wh += dg_power_delivered_w_t * dt

                if dg_rated_w == 0:
                    num_dg_active_this_hour = num_dg
                else:
                    num_dg_active_this_hour = min(math.ceil(dg_power_delivered_w_t / dg_rated_w), num_dg)

        base_fuel_L = (
            num_dg_active_this_hour * (base_load_cons * dg_rated_w * dt)
            if num_dg_active_this_hour > 0 else 0.0
        )
        variable_fuel_L = fuel_cons_factor * dg_power_delivered_w_t * dt
        current_hour_fuel_L = variable_fuel_L + base_fuel_L

        current_hour_fuel_price_cost = current_hour_fuel_L * fuel_price
        current_hour_co2_kg = current_hour_fuel_L * co2_factor

        total_dg_fuel_cost_usd += current_hour_fuel_price_cost
        total_co2_emissions_kg += current_hour_co2_kg

        remaining_load_w -= dg_power_delivered_w_t

        load_shed_w_t = max(0.0, remaining_load_w)
        if load_shed_w_t > 0.0:
            total_load_shedding_cost += load_shed_w_t * cost_load_shed * dt

        curtailed_pv_w = pv_w_t - pv_used_w
        curtailed_wind_w = wind_w_t - wind_used_w
        total_curtailment_w_t = max(0.0, curtailed_pv_w) + max(0.0, curtailed_wind_w)
        if total_curtailment_w_t > 0.0:
            total_curtailment_cost += total_curtailment_w_t * cost_curt * dt

    battery_model.current_soc = current_soc

    total_weekly_op_cost = (
        total_dg_fuel_cost_usd +
        total_dg_om_cost_usd +
        total_battery_degradation_cost +
        total_load_shedding_cost +
        total_curtailment_cost
    )

    return {
        'weekly_op_cost': total_weekly_op_cost,
        'weekly_co2': total_co2_emissions_kg,
        'total_dg_energy_wh': dg_energy_total_wh,
        'total_load_energy_wh': total_load_wh,
        'fuel_cost_usd': total_dg_fuel_cost_usd,
        'battery_deg_cost_usd': total_battery_degradation_cost,
        'curtailment_cost_usd': total_curtailment_cost,
        'load_shedding_cost_usd': total_load_shedding_cost,
        'dg_hours_active': dg_hours_active
    }


def estimate_operational_performance_lp(
    season_df,
    pv_power_available_w,
    wind_power_available_w,
    battery_model,
    dg_model,
    global_params,
    load_demand_w=None
):
    """
    Linear Programming (LP) multi-period look-ahead dispatch baseline using scipy.optimize.linprog (HiGHS).
    Simultaneously optimizes hourly battery charging, battery discharging, DG generation,
    renewable curtailment, and load shedding across the entire horizon to minimize total operational cost.
    After obtaining the LP dispatch trajectory, evaluates the exact nonlinear modular DG step fuel cost
    (ceil(P_dg / P_rated)) as well as the linearized LP objective for fair comparison with greedy EMS.
    """
    dt = global_params['DT']
    if load_demand_w is None:
        load_demand_w = season_df["Load (kW)"].values * 1000.0
    T = len(load_demand_w)

    pv_avail_w = pv_power_available_w * global_params.get('system_eff_pv', 1.0) * 0.98
    wind_avail_w = wind_power_available_w * global_params.get('system_eff_wind', 1.0) * 0.98
    re_avail_w = pv_avail_w + wind_avail_w

    num_batt = battery_model.num_batteries
    batt_cap_wh = num_batt * battery_model.energy_capacity
    conv_eff = battery_model.converter_efficiency
    soc_min = battery_model.soc_min
    soc_max = battery_model.soc_max
    soc_init = (soc_min + soc_max) / 2.0

    max_ch_w = (num_batt * battery_model.max_charge / conv_eff) if conv_eff > 0 else 0.0
    max_dis_w = num_batt * battery_model.max_discharge * conv_eff
    max_dg_w = dg_model.num_generators * dg_model.rated_power

    deg_cost_per_wh = battery_model.capital_cost / (
        battery_model.lifetime_cycles * battery_model.energy_capacity
    )
    # Effective continuous fuel slope in LP (variable + average base load slope)
    eff_fuel_l_per_wh = dg_model.fuel_consumption_factor + dg_model.base_load_consumption
    c_dg = eff_fuel_l_per_wh * dg_model.fuel_price_per_liter * dt
    c_ch = deg_cost_per_wh * conv_eff * dt
    c_dis = (deg_cost_per_wh / conv_eff) * dt
    c_curt = global_params['cost_curtailment_per_wh'] * dt
    c_shed = global_params['cost_load_shed_per_wh'] * dt

    # Decision variables per hour t: [P_re_used_t, P_ch_t, P_dis_t, P_dg_t, P_curt_t, P_shed_t] -> 6*T vars
    n_vars = 6 * T
    c = np.zeros(n_vars)
    bounds = []
    for t in range(T):
        c[6*t + 0] = 0.0     # P_re_used (direct to load)
        c[6*t + 1] = c_ch    # P_ch
        c[6*t + 2] = c_dis   # P_dis
        c[6*t + 3] = c_dg    # P_dg
        c[6*t + 4] = c_curt  # P_curt
        c[6*t + 5] = c_shed  # P_shed
        bounds.extend([
            (0.0, float(re_avail_w[t])),
            (0.0, float(max_ch_w)),
            (0.0, float(max_dis_w)),
            (0.0, float(max_dg_w)),
            (0.0, float(re_avail_w[t])),
            (0.0, float(load_demand_w[t]))
        ])

    # Equality constraints:
    # 1) Load balance at each t: P_re_used_t + P_dis_t + P_dg_t + P_shed_t = Load_t
    # 2) Renewable split at each t: P_re_used_t + P_ch_t + P_curt_t = RE_avail_t
    A_eq = np.zeros((2 * T, n_vars))
    b_eq = np.zeros(2 * T)
    for t in range(T):
        # Load balance
        A_eq[t, 6*t + 0] = 1.0
        A_eq[t, 6*t + 2] = 1.0
        A_eq[t, 6*t + 3] = 1.0
        A_eq[t, 6*t + 5] = 1.0
        b_eq[t] = load_demand_w[t]
        # Renewable balance
        A_eq[T + t, 6*t + 0] = 1.0
        A_eq[T + t, 6*t + 1] = 1.0
        A_eq[T + t, 6*t + 4] = 1.0
        b_eq[T + t] = re_avail_w[t]

    # Inequality constraints: SoC_min <= SoC_t <= SoC_max for t = 0..T-1
    # E_t = E_0 + sum_{k=0}^t (P_ch_k * conv_eff - P_dis_k / conv_eff) * dt
    A_ub = np.zeros((2 * T, n_vars))
    b_ub = np.zeros(2 * T)
    e_init = soc_init * batt_cap_wh
    e_max = soc_max * batt_cap_wh
    e_min = soc_min * batt_cap_wh
    for t in range(T):
        for k in range(t + 1):
            # Upper bound: sum(ch*eff - dis/eff)*dt <= e_max - e_init
            A_ub[t, 6*k + 1] = conv_eff * dt
            A_ub[t, 6*k + 2] = -dt / conv_eff
            # Lower bound: -sum(ch*eff - dis/eff)*dt <= e_init - e_min
            A_ub[T + t, 6*k + 1] = -conv_eff * dt
            A_ub[T + t, 6*k + 2] = dt / conv_eff
        b_ub[t] = e_max - e_init
        b_ub[T + t] = e_init - e_min

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Post-evaluate LP schedule under exact modular DG fuel equation & degradation equation
    x_sol = res.x
    total_dg_fuel_cost_usd = 0.0
    total_battery_degradation_cost = 0.0
    total_curtailment_cost = 0.0
    total_load_shedding_cost = 0.0
    total_co2_emissions_kg = 0.0
    dg_energy_total_wh = 0.0
    total_load_wh = float(np.sum(load_demand_w) * dt)

    for t in range(T):
        p_ch = max(0.0, x_sol[6*t + 1])
        p_dis = max(0.0, x_sol[6*t + 2])
        p_dg = max(0.0, x_sol[6*t + 3])
        p_curt = max(0.0, x_sol[6*t + 4])
        p_shed = max(0.0, x_sol[6*t + 5])

        e_ch_wh = p_ch * conv_eff * dt
        e_dis_wh = (p_dis / conv_eff) * dt
        total_battery_degradation_cost += (e_ch_wh + e_dis_wh) * deg_cost_per_wh

        num_active = 0
        if p_dg > 1e-3 and dg_model.num_generators > 0:
            dg_energy_total_wh += p_dg * dt
            num_active = min(math.ceil(p_dg / dg_model.rated_power), dg_model.num_generators)

        base_fuel_L = num_active * (dg_model.base_load_consumption * dg_model.rated_power * dt)
        var_fuel_L = dg_model.fuel_consumption_factor * p_dg * dt
        fuel_L = base_fuel_L + var_fuel_L
        total_dg_fuel_cost_usd += fuel_L * dg_model.fuel_price_per_liter
        total_co2_emissions_kg += fuel_L * dg_model.co2_emission_factor * dg_model.diesel_cfactor
        total_curtailment_cost += p_curt * global_params['cost_curtailment_per_wh'] * dt
        total_load_shedding_cost += p_shed * global_params['cost_load_shed_per_wh'] * dt

    total_weekly_op_cost = (
        total_dg_fuel_cost_usd +
        total_battery_degradation_cost +
        total_curtailment_cost +
        total_load_shedding_cost
    )

    return {
        'weekly_op_cost': total_weekly_op_cost,
        'weekly_op_cost_linear_lp': float(res.fun),
        'weekly_co2': total_co2_emissions_kg,
        'total_dg_energy_wh': dg_energy_total_wh,
        'total_load_energy_wh': total_load_wh,
        'fuel_cost_usd': total_dg_fuel_cost_usd,
        'battery_deg_cost_usd': total_battery_degradation_cost,
        'curtailment_cost_usd': total_curtailment_cost,
        'load_shedding_cost_usd': total_load_shedding_cost
    }
