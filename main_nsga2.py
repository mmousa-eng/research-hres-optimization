import os
import sys
import numpy as np
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from size_nsga2 import (
    run_annual_nsga2_optimization,
    DEFAULT_GLOBAL_EMS_PARAMS,
    PV_MODULE_RATING_W,
    WIND_TURBINE_RATING_W,
    BATTERY_MODULE_ENERGY_WH,
    DG_UNIT_RATING_W
)
from utils.load_data import load_data


def compute_lifecycle_economics(
    num_pv,
    num_wind,
    num_batt,
    num_dg,
    annual_op_cost_usd,
    annual_load_kwh,
    project_lifetime_yrs=20,
    discount_rate=0.06,
    pv_cap_cost=7000.0,
    wind_cap_cost=38745.0,
    batt_cap_cost=2000.0,
    dg_cap_cost=1514.0,
    pv_om_yr=20.0,
    wind_om_pct=0.03,
    batt_om_pct=0.05,
    batt_replacement_yr=10,
    include_fixed_om_in_npc=True,
    deduct_cycle_deg_when_explicit_replacement=False,
    annual_batt_deg_cost_usd=0.0
):
    """
    Computes standard lifecycle economic metrics (NPC, CRF, TAC, and LCOE) over the project
    horizon (default 20 years) at the specified real discount rate (default 6%), accounting
    for initial capital expenditure, discounted battery replacement at year 10, and discounted
    annual operational & maintenance (O&M) cash flows.
    """
    # 1. Initial Capital Cost at t = 0
    cap_pv = num_pv * pv_cap_cost
    cap_wind = num_wind * wind_cap_cost
    cap_batt = num_batt * batt_cap_cost
    cap_dg = num_dg * dg_cap_cost
    initial_capital_cost = cap_pv + cap_wind + cap_batt + cap_dg

    # 2. Capital Recovery Factor (CRF) and Present Worth Factor (PWF)
    if discount_rate > 0:
        crf = (discount_rate * (1.0 + discount_rate) ** project_lifetime_yrs) / (
            (1.0 + discount_rate) ** project_lifetime_yrs - 1.0
        )
        pwf = 1.0 / crf
    else:
        crf = 1.0 / project_lifetime_yrs
        pwf = float(project_lifetime_yrs)

    # 3. Present Value of Battery Replacement (at year 10 within 20-year project lifetime)
    if 0 < batt_replacement_yr < project_lifetime_yrs:
        pv_replacement_cost = cap_batt / ((1.0 + discount_rate) ** batt_replacement_yr)
    else:
        pv_replacement_cost = 0.0

    # 4. Annual Fixed O&M + Variable Operational Cost
    fixed_om_annual = (
        num_pv * pv_om_yr +
        num_wind * (wind_om_pct * wind_cap_cost) +
        num_batt * (batt_om_pct * batt_cap_cost)
    )
    eff_op_cost = (
        annual_op_cost_usd - annual_batt_deg_cost_usd
        if deduct_cycle_deg_when_explicit_replacement else annual_op_cost_usd
    )
    total_annual_om = eff_op_cost + (fixed_om_annual if include_fixed_om_in_npc else 0.0)
    pv_om_cost = total_annual_om * pwf

    # 5. Net Present Cost (NPC), Total Annualized Cost (TAC), and LCOE ($/kWh)
    npc = initial_capital_cost + pv_replacement_cost + pv_om_cost
    annualized_total_cost = npc * crf
    lcoe_usd_per_kwh = annualized_total_cost / annual_load_kwh if annual_load_kwh > 0 else float('inf')

    return {
        'initial_capital_cost_usd': initial_capital_cost,
        'pv_replacement_cost_usd': pv_replacement_cost,
        'annual_fixed_om_usd': fixed_om_annual,
        'annual_operational_cost_usd': annual_op_cost_usd,
        'total_annual_om_usd': total_annual_om,
        'pv_om_cost_usd': pv_om_cost,
        'npc_usd': npc,
        'crf': crf,
        'pwf': pwf,
        'annualized_total_cost_usd': annualized_total_cost,
        'lcoe_usd_per_kwh': lcoe_usd_per_kwh
    }


def select_topsis_solution(results, weights=(1/3, 1/3, 1/3)):
    """
    Selects the best compromise solution from the Pareto front using TOPSIS
    (Technique for Order of Preference by Similarity to Ideal Solution).
    """
    F = np.array([
        [r['investment_cost'], r['operational_cost'], r['co2_emissions']]
        for r in results
    ], dtype=float)
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)

    # Vector normalization
    norms = np.linalg.norm(F, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    R = F / norms
    V = R * w

    # Ideal best (minimum for cost/emissions) and ideal worst (maximum)
    v_plus = np.min(V, axis=0)
    v_minus = np.max(V, axis=0)

    d_plus = np.sqrt(np.sum((V - v_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((V - v_minus) ** 2, axis=1))
    closeness = d_minus / (d_plus + d_minus + 1e-12)

    best_idx = int(np.argmax(closeness))
    return results[best_idx], closeness, best_idx


def evaluate_all_seasons_summary(num_pv, num_wind, num_batt, num_dg, seasonal_data, ems_params):
    """
    Evaluates a given integer component configuration across all 4 seasonal datasets,
    properly weighted by (52 / len(seasonal_data)) weeks per year.
    """
    from ems import estimate_operational_performance
    from models.pv_model import PVModel
    from models.wind_model import WindModel
    from models.batteries_model import BatteryModel
    from models.dg_model import DGModel

    total_dg_wh = 0.0
    total_load_wh = 0.0
    total_op_cost = 0.0
    total_co2 = 0.0
    total_fuel_cost = 0.0
    total_batt_deg_cost = 0.0
    total_curt_cost = 0.0
    total_shed_cost = 0.0
    scale = 52.0 / len(seasonal_data)

    for season_name, df in seasonal_data.items():
        pv_model = PVModel(
            num_modules=num_pv,
            G=df["G(W/m^2)"].values,
            Ta=df["Ta(C) at 2m"].values,
            V=df["V(m/s) at 10m"].values
        )
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

        total_op_cost += result['weekly_op_cost'] * scale
        total_co2 += result['weekly_co2'] * scale
        total_dg_wh += result['total_dg_energy_wh'] * scale
        total_load_wh += result['total_load_energy_wh'] * scale
        total_fuel_cost += result.get('fuel_cost_usd', 0.0) * scale
        total_batt_deg_cost += result.get('battery_deg_cost_usd', 0.0) * scale
        total_curt_cost += result.get('curtailment_cost_usd', 0.0) * scale
        total_shed_cost += result.get('load_shedding_cost_usd', 0.0) * scale

    dg_contribution_pct = (100.0 * total_dg_wh / total_load_wh) if total_load_wh > 0 else 0.0
    renewable_fraction_pct = 100.0 - dg_contribution_pct

    return {
        'annual_op_cost': total_op_cost,
        'annual_co2': total_co2,
        'annual_dg_kwh': total_dg_wh / 1000.0,
        'annual_load_kwh': total_load_wh / 1000.0,
        'dg_contribution_pct': dg_contribution_pct,
        'renewable_fraction_pct': renewable_fraction_pct,
        'annual_fuel_cost': total_fuel_cost,
        'annual_batt_deg_cost': total_batt_deg_cost,
        'annual_curt_cost': total_curt_cost,
        'annual_shed_cost': total_shed_cost
    }


def print_solution_summary(solution, seasonal_data, ems_params):
    pv_kw = solution['pv_kw']
    wind_kw = solution['wind_kw']
    batt_kwh = solution['battery_kwh']
    dg_kw = solution['dg_kw']

    # Use math.ceil consistent with size_nsga2.py
    num_pv = max(1, math.ceil(pv_kw * 1000.0 / PV_MODULE_RATING_W))
    num_wind = max(0, math.ceil(wind_kw * 1000.0 / WIND_TURBINE_RATING_W))
    num_batt = max(1, math.ceil(batt_kwh * 1000.0 / BATTERY_MODULE_ENERGY_WH))
    num_dg = max(1, math.ceil(dg_kw * 1000.0 / DG_UNIT_RATING_W))

    perf = evaluate_all_seasons_summary(num_pv, num_wind, num_batt, num_dg, seasonal_data, ems_params)
    econ = compute_lifecycle_economics(
        num_pv, num_wind, num_batt, num_dg,
        annual_op_cost_usd=solution['operational_cost'],
        annual_load_kwh=perf['annual_load_kwh'],
        project_lifetime_yrs=20,
        discount_rate=0.06
    )

    print("\nSolution Summary:")
    print(f"  PV Capacity: {pv_kw:.2f} kW ({num_pv} modules)")
    print(f"  Wind Capacity: {wind_kw:.2f} kW ({num_wind} turbines)")
    print(f"  Battery Capacity: {batt_kwh:.2f} kWh ({num_batt} modules)")
    print(f"  DG Capacity: {dg_kw:.2f} kW ({num_dg} units)")
    print(f"  Annualized Investment Objective (f1): ${solution['investment_cost']:,.2f} / year")
    print(f"  Annual Operational Objective (f2): ${solution['operational_cost']:,.2f} / year")
    print(f"  Annual CO2 Emissions (f3): {solution['co2_emissions']:,.2f} kg / year")
    print(f"  Initial Capital Cost (CapEx): ${econ['initial_capital_cost_usd']:,.2f}")
    print(f"  20-Year Net Present Cost (NPC @ 6%): ${econ['npc_usd']:,.2f}")
    print(f"  Levelized Cost of Energy (LCOE @ 6%): ${econ['lcoe_usd_per_kwh']:.4f} / kWh")
    print(f"  DG Contribution to Load (all 4 seasons): {perf['dg_contribution_pct']:.2f}%")
    print(f"  Renewable Energy Fraction (RF): {perf['renewable_fraction_pct']:.2f}%")


def print_key_solutions(results, seasonal_data, ems_params):
    min_inv_sol = min(results, key=lambda x: x['investment_cost'])
    min_op_sol = min(results, key=lambda x: x['operational_cost'])
    min_co2_sol = min(results, key=lambda x: x['co2_emissions'])

    # Min-Max Normalization with Equal Weighting
    inv_vals = np.array([r['investment_cost'] for r in results])
    op_vals = np.array([r['operational_cost'] for r in results])
    co2_vals = np.array([r['co2_emissions'] for r in results])

    inv_norm = (inv_vals - inv_vals.min()) / (inv_vals.max() - inv_vals.min() + 1e-9)
    op_norm = (op_vals - op_vals.min()) / (op_vals.max() - op_vals.min() + 1e-9)
    co2_norm = (co2_vals - co2_vals.min()) / (co2_vals.max() - co2_vals.min() + 1e-9)

    combined_scores = inv_norm + op_norm + co2_norm
    best_combined_sol = results[int(np.argmin(combined_scores))]

    # TOPSIS MCDM Selection
    topsis_sol, topsis_scores, topsis_idx = select_topsis_solution(results)

    print("\n--- Key Pareto-Optimal HRES Configurations ---")
    print("\n[1] Minimum Investment Cost Solution")
    print_solution_summary(min_inv_sol, seasonal_data, ems_params)

    print("\n[2] Minimum Operational Cost Solution")
    print_solution_summary(min_op_sol, seasonal_data, ems_params)

    print("\n[3] Minimum Emissions Solution")
    print_solution_summary(min_co2_sol, seasonal_data, ems_params)

    print("\n[4] Best Combined Trade-off Solution (Min-Max Equal Weighting)")
    print_solution_summary(best_combined_sol, seasonal_data, ems_params)

    print(f"\n[5] TOPSIS Selected Solution (Closeness Score = {topsis_scores[topsis_idx]:.4f})")
    print_solution_summary(topsis_sol, seasonal_data, ems_params)


if __name__ == "__main__":
    seasonal_data = load_data()
    ems_params = DEFAULT_GLOBAL_EMS_PARAMS.copy()
    results = run_annual_nsga2_optimization(seasonal_data, ems_params)
    print_key_solutions(results, seasonal_data, ems_params)
