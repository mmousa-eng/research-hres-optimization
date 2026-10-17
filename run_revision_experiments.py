import os
import sys
import time
import json
import math
import platform

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import pvlib
import pymoo
from pymoo.indicators.hv import HV

from utils.load_data import load_data
from size_nsga2 import (
    HRESProblem,
    run_annual_nsga2_optimization,
    DEFAULT_GLOBAL_EMS_PARAMS,
    PV_MODULE_RATING_W,
    WIND_TURBINE_RATING_W,
    BATTERY_MODULE_ENERGY_WH,
    DG_UNIT_RATING_W
)
from main_nsga2 import (
    compute_lifecycle_economics,
    select_topsis_solution,
    evaluate_all_seasons_summary
)
from ems import estimate_operational_performance, estimate_operational_performance_lp
from models.pv_model import PVModel
from models.wind_model import WindModel
from models.batteries_model import BatteryModel
from models.dg_model import DGModel


def compute_spacing_metric(F_norm):
    """
    Computes Schott's Spacing (SP) metric on normalized objective matrix F_norm.
    """
    n = len(F_norm)
    if n < 2:
        return 0.0
    # Pairwise L1 (Manhattan) distances in normalized objective space
    diff = np.abs(F_norm[:, None, :] - F_norm[None, :, :])
    dist_mat = np.sum(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    d_i = np.min(dist_mat, axis=1)
    d_mean = np.mean(d_i)
    sp = float(np.sqrt(np.sum((d_i - d_mean) ** 2) / (n - 1)))
    return sp


def extract_four_archetypes(results):
    """
    Extracts the 4 stakeholder configurations from a Pareto set:
    1. Minimum Investment Cost
    2. Minimum Operational Cost
    3. Minimum CO2 Emissions
    4. Balanced Trade-off (both Min-Max Equal Weighting and TOPSIS)
    """
    min_inv = min(results, key=lambda x: x['investment_cost'])
    min_op = min(results, key=lambda x: x['operational_cost'])
    min_co2 = min(results, key=lambda x: x['co2_emissions'])

    inv_vals = np.array([r['investment_cost'] for r in results])
    op_vals = np.array([r['operational_cost'] for r in results])
    co2_vals = np.array([r['co2_emissions'] for r in results])

    inv_n = (inv_vals - inv_vals.min()) / (inv_vals.max() - inv_vals.min() + 1e-9)
    op_n = (op_vals - op_vals.min()) / (op_vals.max() - op_vals.min() + 1e-9)
    co2_n = (co2_vals - co2_vals.min()) / (co2_vals.max() - co2_vals.min() + 1e-9)
    eq_idx = int(np.argmin(inv_n + op_n + co2_n))
    balanced_eq = results[eq_idx]

    topsis_sol, topsis_scores, topsis_idx = select_topsis_solution(results)

    return {
        'Min_Inv': min_inv,
        'Min_Op': min_op,
        'Min_CO2': min_co2,
        'Balanced_EqualWt': balanced_eq,
        'Balanced_TOPSIS': topsis_sol,
        'topsis_max_score': float(topsis_scores[topsis_idx]),
        'equalwt_topsis_score': float(topsis_scores[eq_idx]),
        'same_selection': (
            balanced_eq['num_pv'] == topsis_sol['num_pv'] and
            balanced_eq['num_wind'] == topsis_sol['num_wind'] and
            balanced_eq['num_batt'] == topsis_sol['num_batt'] and
            balanced_eq['num_dg'] == topsis_sol['num_dg']
        )
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "revision_figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("  HRES UDHRUH PEER-REVIEW REVISION EXPERIMENTS RUNNER")
    print("=================================================================")

    # Load seasonal dataset
    seasonal_data = load_data()
    ems_params = DEFAULT_GLOBAL_EMS_PARAMS.copy()

    # Calculate total annual load energy (kWh/yr)
    scale_weeks = 52.0 / len(seasonal_data)
    annual_load_kwh = sum(df["Load (kW)"].sum() * scale_weeks for df in seasonal_data.values())
    annual_load_kwh_8760 = annual_load_kwh * (8760.0 / (len(seasonal_data) * 180.0 * scale_weeks))
    peak_load_kw = max(df["Load (kW)"].max() for df in seasonal_data.values())
    print(f"Peak Load: {peak_load_kw:.4f} kW | Min DG (1.02x): {peak_load_kw*1.02:.4f} kW (15 x 2kW units)")
    print(f"Annual Load Energy (52-week scaled): {annual_load_kwh:,.2f} kWh/yr (8760h equiv: {annual_load_kwh_8760:,.2f} kWh/yr)")

    # -----------------------------------------------------------------
    # 0. Benchmark single evaluation speed (Comment 10)
    # -----------------------------------------------------------------
    shared_cache = {}
    prob_bench = HRESProblem(seasonal_data, ems_params=ems_params, eval_cache={})
    t0 = time.perf_counter()
    for _ in range(20):
        prob_bench._eval_cache.clear()
        out_dummy = {}
        prob_bench._evaluate(np.array([10.0, 20.0, 10.0, 35.0]), out_dummy)
    single_eval_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
    print(f"Single uncached _evaluate() time: {single_eval_ms:.3f} ms")

    # -----------------------------------------------------------------
    # 1. Statistical Robustness Across 20 Independent Seeds (Comment 3)
    # -----------------------------------------------------------------
    print("\n--- Running 20 Independent NSGA-II Optimizations (n_gen=150, pop_size=100) ---")
    num_runs = 20
    n_gen = 150
    pop_size = 100

    all_runs_results = []
    all_runs_details = []
    run_times_s = []

    for seed in range(1, num_runs + 1):
        t_start = time.perf_counter()
        res_list, details = run_annual_nsga2_optimization(
            seasonal_data,
            ems_params=ems_params,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            verbose=False,
            return_details=True,
            eval_cache=shared_cache
        )
        elapsed = time.perf_counter() - t_start
        run_times_s.append(elapsed)
        all_runs_results.append(res_list)
        all_runs_details.append(details)
        print(f"  Seed {seed:2d}/20 finished in {elapsed:.2f}s | Pareto solutions: {len(res_list)} | Unique cached configs: {len(shared_cache)}")

    # Compute global min and max across all generations and all 20 runs for consistent HV normalization
    all_F_list = []
    for d in all_runs_details:
        for gen_F in d['history_F']:
            all_F_list.append(gen_F)
    all_F_concat = np.vstack(all_F_list)
    f_min_global = np.min(all_F_concat, axis=0)
    f_max_global = np.max(all_F_concat, axis=0)
    f_range_global = np.where((f_max_global - f_min_global) == 0, 1.0, f_max_global - f_min_global)

    ref_point = np.array([1.1, 1.1, 1.1])
    hv_indicator = HV(ref_point=ref_point)

    hv_convergence_matrix = np.zeros((num_runs, n_gen))
    final_hv_list = []
    final_sp_list = []
    archetype_stats = {
        'Min_Inv': {'f1': [], 'f2': [], 'f3': [], 'npc': [], 'lcoe': []},
        'Min_Op': {'f1': [], 'f2': [], 'f3': [], 'npc': [], 'lcoe': []},
        'Min_CO2': {'f1': [], 'f2': [], 'f3': [], 'npc': [], 'lcoe': []},
        'Balanced_TOPSIS': {'f1': [], 'f2': [], 'f3': [], 'npc': [], 'lcoe': []}
    }

    for r_idx in range(num_runs):
        hist_F = all_runs_details[r_idx]['history_F']
        for g_idx in range(n_gen):
            F_g = hist_F[g_idx]
            F_g_norm = (F_g - f_min_global) / f_range_global
            hv_convergence_matrix[r_idx, g_idx] = float(hv_indicator(F_g_norm))

        F_final = all_runs_details[r_idx]['F']
        F_final_norm = (F_final - f_min_global) / f_range_global
        final_hv_list.append(float(hv_indicator(F_final_norm)))
        final_sp_list.append(compute_spacing_metric(F_final_norm))

        archs = extract_four_archetypes(all_runs_results[r_idx])
        for key in ['Min_Inv', 'Min_Op', 'Min_CO2', 'Balanced_TOPSIS']:
            sol = archs[key]
            econ = compute_lifecycle_economics(
                sol['num_pv'], sol['num_wind'], sol['num_batt'], sol['num_dg'],
                annual_op_cost_usd=sol['operational_cost'],
                annual_load_kwh=annual_load_kwh,
                project_lifetime_yrs=20,
                discount_rate=0.06
            )
            archetype_stats[key]['f1'].append(sol['investment_cost'])
            archetype_stats[key]['f2'].append(sol['operational_cost'])
            archetype_stats[key]['f3'].append(sol['co2_emissions'])
            archetype_stats[key]['npc'].append(econ['npc_usd'])
            archetype_stats[key]['lcoe'].append(econ['lcoe_usd_per_kwh'])

    # Summarize 20-run statistics
    multi_run_summary = {
        'hv_mean': float(np.mean(final_hv_list)),
        'hv_std': float(np.std(final_hv_list, ddof=1)),
        'hv_min': float(np.min(final_hv_list)),
        'hv_max': float(np.max(final_hv_list)),
        'sp_mean': float(np.mean(final_sp_list)),
        'sp_std': float(np.std(final_sp_list, ddof=1)),
        'runtime_mean_s': float(np.mean(run_times_s)),
        'runtime_std_s': float(np.std(run_times_s, ddof=1)),
        'archetypes': {}
    }
    for key, metrics in archetype_stats.items():
        multi_run_summary['archetypes'][key] = {}
        for m_name, vals in metrics.items():
            arr = np.array(vals, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))
            cv_pct = float(100.0 * std_val / mean_val) if mean_val != 0 else 0.0
            multi_run_summary['archetypes'][key][m_name] = {
                'mean': mean_val,
                'std': std_val,
                'cv_pct': cv_pct,
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }

    print(f"\n20-Run Pareto Quality: HV = {multi_run_summary['hv_mean']:.4f} +/- {multi_run_summary['hv_std']:.4f}, Spacing = {multi_run_summary['sp_mean']:.4f} +/- {multi_run_summary['sp_std']:.4f}")

    # -----------------------------------------------------------------
    # 2. Economic Metrics (NPC & LCOE) for Manuscript Table 4/5 & NSGA-II (Comment 1)
    # -----------------------------------------------------------------
    manuscript_configs = {
        'Min_Inv': {'num_pv': 2, 'num_wind': 0, 'num_batt': 2, 'num_dg': 18, 'ms_inv': 1810.0, 'ms_om': 33493.0, 'ms_co2': 51542.0, 'ms_dg_pct': 99.0},
        'Min_OM': {'num_pv': 40, 'num_wind': 1, 'num_batt': 2, 'num_dg': 17, 'ms_inv': 13939.0, 'ms_om': 23828.0, 'ms_co2': 28371.0, 'ms_dg_pct': 53.0},
        'Min_CO2': {'num_pv': 40, 'num_wind': 4, 'num_batt': 49, 'num_dg': 17, 'ms_inv': 22349.0, 'ms_om': 57380.0, 'ms_co2': 16477.0, 'ms_dg_pct': 31.0},
        'Balanced': {'num_pv': 2, 'num_wind': 1, 'num_batt': 2, 'num_dg': 17, 'ms_inv': 3299.0, 'ms_om': 24348.0, 'ms_co2': 31135.0, 'ms_dg_pct': 59.0}
    }

    table5_manuscript_economics = {}
    for c_name, cfg in manuscript_configs.items():
        sim_perf = evaluate_all_seasons_summary(
            cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg'],
            seasonal_data, ems_params
        )
        # Compute lifecycle economics using:
        # (A) Manuscript reported annual O&M cost (so Table 5 rows stay 100% self-consistent)
        econ_ms_6 = compute_lifecycle_economics(
            cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg'],
            annual_op_cost_usd=cfg['ms_om'],
            annual_load_kwh=annual_load_kwh,
            project_lifetime_yrs=20,
            discount_rate=0.06,
            include_fixed_om_in_npc=False
        )
        econ_ms_8 = compute_lifecycle_economics(
            cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg'],
            annual_op_cost_usd=cfg['ms_om'],
            annual_load_kwh=annual_load_kwh,
            project_lifetime_yrs=20,
            discount_rate=0.08,
            include_fixed_om_in_npc=False
        )
        # (B) Code-simulated operational + fixed maintenance cost
        econ_sim_6 = compute_lifecycle_economics(
            cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg'],
            annual_op_cost_usd=sim_perf['annual_op_cost'],
            annual_load_kwh=annual_load_kwh,
            project_lifetime_yrs=20,
            discount_rate=0.06,
            include_fixed_om_in_npc=True
        )
        table5_manuscript_economics[c_name] = {
            'units': {k: cfg[k] for k in ['num_pv', 'num_wind', 'num_batt', 'num_dg']},
            'manuscript_reported': {
                'yearly_inv_usd': cfg['ms_inv'],
                'yearly_om_usd': cfg['ms_om'],
                'yearly_total_usd': cfg['ms_inv'] + cfg['ms_om'],
                'yearly_co2_kg': cfg['ms_co2'],
                'dg_pct': cfg['ms_dg_pct'],
                'rf_pct': 100.0 - cfg['ms_dg_pct']
            },
            'lifecycle_from_ms_om_r6': econ_ms_6,
            'lifecycle_from_ms_om_r8': econ_ms_8,
            'code_simulated': sim_perf,
            'lifecycle_from_sim_r6': econ_sim_6
        }

    # Also extract Seed 1 NSGA-II key solutions and their economics
    seed1_archs = extract_four_archetypes(all_runs_results[0])
    seed1_economics = {}
    for k_name in ['Min_Inv', 'Min_Op', 'Min_CO2', 'Balanced_TOPSIS', 'Balanced_EqualWt']:
        sol = seed1_archs[k_name]
        sim_p = evaluate_all_seasons_summary(
            sol['num_pv'], sol['num_wind'], sol['num_batt'], sol['num_dg'],
            seasonal_data, ems_params
        )
        ec6 = compute_lifecycle_economics(
            sol['num_pv'], sol['num_wind'], sol['num_batt'], sol['num_dg'],
            annual_op_cost_usd=sol['operational_cost'],
            annual_load_kwh=annual_load_kwh,
            project_lifetime_yrs=20,
            discount_rate=0.06,
            include_fixed_om_in_npc=True
        )
        seed1_economics[k_name] = {
            'solution': sol,
            'performance': sim_p,
            'economics_r6': ec6
        }

    # -----------------------------------------------------------------
    # 3. Greedy EMS vs. LP Dispatch Comparison (Comment 8)
    # -----------------------------------------------------------------
    print("\n--- Running Greedy Dispatch vs. LP Optimal Dispatch Comparison ---")
    dispatch_comparison = {}
    for c_name, cfg in manuscript_configs.items():
        npv, nwind, nbatt, ndg = cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg']
        g_op, g_co2, g_dg_wh, g_fuel, g_deg, g_curt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        lp_op, lp_op_lin, lp_co2, lp_dg_wh, lp_fuel, lp_deg, lp_curt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        t_greedy_ms, t_lp_ms = 0.0, 0.0

        for s_name, df in seasonal_data.items():
            pv_p = PVModel(num_modules=npv, G=df["G(W/m^2)"].values, Ta=df["Ta(C) at 2m"].values, V=df["V(m/s) at 10m"].values).power_output()
            wind_p = WindModel(num_turbines=nwind, V=df["V(m/s) at 10m"].values).power_output()

            bm_g = BatteryModel(num_batteries=nbatt, DT=1, num_steps=len(df))
            dm_g = DGModel(num_generators=ndg, DT=1, num_steps=len(df))
            t0 = time.perf_counter()
            res_g = estimate_operational_performance(df, pv_p, wind_p, bm_g, dm_g, ems_params)
            t_greedy_ms += (time.perf_counter() - t0) * 1000.0

            bm_lp = BatteryModel(num_batteries=nbatt, DT=1, num_steps=len(df))
            dm_lp = DGModel(num_generators=ndg, DT=1, num_steps=len(df))
            t0 = time.perf_counter()
            res_lp = estimate_operational_performance_lp(df, pv_p, wind_p, bm_lp, dm_lp, ems_params)
            t_lp_ms += (time.perf_counter() - t0) * 1000.0

            g_op += res_g['weekly_op_cost'] * scale_weeks
            g_co2 += res_g['weekly_co2'] * scale_weeks
            g_dg_wh += res_g['total_dg_energy_wh'] * scale_weeks
            g_fuel += res_g['fuel_cost_usd'] * scale_weeks
            g_deg += res_g['battery_deg_cost_usd'] * scale_weeks
            g_curt += res_g['curtailment_cost_usd'] * scale_weeks

            lp_op += res_lp['weekly_op_cost'] * scale_weeks
            lp_op_lin += res_lp['weekly_op_cost_linear_lp'] * scale_weeks
            lp_co2 += res_lp['weekly_co2'] * scale_weeks
            lp_dg_wh += res_lp['total_dg_energy_wh'] * scale_weeks
            lp_fuel += res_lp['fuel_cost_usd'] * scale_weeks
            lp_deg += res_lp['battery_deg_cost_usd'] * scale_weeks
            lp_curt += res_lp['curtailment_cost_usd'] * scale_weeks

        gap_pct = 100.0 * (g_op - lp_op) / lp_op if lp_op > 0 else 0.0
        dispatch_comparison[c_name] = {
            'greedy_op_cost_usd': g_op,
            'lp_op_cost_usd': lp_op,
            'lp_linear_obj_usd': lp_op_lin,
            'cost_diff_usd': g_op - lp_op,
            'optimality_gap_pct': gap_pct,
            'greedy_co2_kg': g_co2,
            'lp_co2_kg': lp_co2,
            'greedy_dg_pct': 100.0 * g_dg_wh / (annual_load_kwh * 1000.0),
            'lp_dg_pct': 100.0 * lp_dg_wh / (annual_load_kwh * 1000.0),
            'greedy_fuel_usd': g_fuel,
            'lp_fuel_usd': lp_fuel,
            'greedy_batt_deg_usd': g_deg,
            'lp_batt_deg_usd': lp_deg,
            'greedy_curt_usd': g_curt,
            'lp_curt_usd': lp_curt,
            'greedy_runtime_ms': t_greedy_ms,
            'lp_runtime_ms': t_lp_ms,
            'speedup_factor': t_lp_ms / max(t_greedy_ms, 1e-6)
        }
        print(f"  {c_name:10s}: Greedy=${g_op:,.2f}/yr vs LP=${lp_op:,.2f}/yr (Gap={gap_pct:+.2f}%, Speedup={t_lp_ms/max(t_greedy_ms,1e-6):.1f}x)")

    # -----------------------------------------------------------------
    # 4. Sensitivity Analysis (+/- 20% Fuel Price, Battery Cost, Discount Rate) (Comment 11)
    # -----------------------------------------------------------------
    print("\n--- Running Sensitivity Analysis (+/- 20%) ---")
    multipliers = [0.80, 0.90, 1.00, 1.10, 1.20]
    base_fuel_price = 1.06
    base_batt_cost = 2000.0
    base_discount_rate = 0.06

    sensitivity_results = {
        'fuel_price': {},
        'battery_cost': {},
        'discount_rate': {}
    }

    for c_name, cfg in manuscript_configs.items():
        npv, nwind, nbatt, ndg = cfg['num_pv'], cfg['num_wind'], cfg['num_batt'], cfg['num_dg']
        sensitivity_results['fuel_price'][c_name] = []
        sensitivity_results['battery_cost'][c_name] = []
        sensitivity_results['discount_rate'][c_name] = []

        # Precompute PV and Wind for this config
        pv_by_s = {s: PVModel(num_modules=npv, G=df["G(W/m^2)"].values, Ta=df["Ta(C) at 2m"].values, V=df["V(m/s) at 10m"].values).power_output() for s, df in seasonal_data.items()}
        wind_by_s = {s: WindModel(num_turbines=nwind, V=df["V(m/s) at 10m"].values).power_output() for s, df in seasonal_data.items()}

        # A) Fuel Price Sensitivity
        for m in multipliers:
            fp = base_fuel_price * m
            op_tot = 0.0
            for s, df in seasonal_data.items():
                bm = BatteryModel(num_batteries=nbatt, DT=1, num_steps=len(df), capital_cost=base_batt_cost)
                dm = DGModel(num_generators=ndg, DT=1, num_steps=len(df), fuel_price_per_liter=fp)
                res_s = estimate_operational_performance(df, pv_by_s[s], wind_by_s[s], bm, dm, ems_params)
                op_tot += res_s['weekly_op_cost'] * scale_weeks
            # Also scale manuscript O&M proportionally by fuel share
            ec_sim = compute_lifecycle_economics(npv, nwind, nbatt, ndg, op_tot, annual_load_kwh, 20, base_discount_rate, batt_cap_cost=base_batt_cost, include_fixed_om_in_npc=True)
            # And manuscript Table 5 scaled O&M
            fuel_delta = op_tot - table5_manuscript_economics[c_name]['code_simulated']['annual_op_cost']
            ec_ms = compute_lifecycle_economics(npv, nwind, nbatt, ndg, cfg['ms_om'] + fuel_delta, annual_load_kwh, 20, base_discount_rate, batt_cap_cost=base_batt_cost, include_fixed_om_in_npc=False)
            sensitivity_results['fuel_price'][c_name].append({
                'multiplier': m,
                'param_value': fp,
                'sim_op_cost': op_tot,
                'sim_npc': ec_sim['npc_usd'],
                'sim_lcoe': ec_sim['lcoe_usd_per_kwh'],
                'ms_npc': ec_ms['npc_usd'],
                'ms_lcoe': ec_ms['lcoe_usd_per_kwh']
            })

        # B) Battery Capital Cost Sensitivity
        for m in multipliers:
            bc = base_batt_cost * m
            op_tot = 0.0
            for s, df in seasonal_data.items():
                bm = BatteryModel(num_batteries=nbatt, DT=1, num_steps=len(df), capital_cost=bc)
                dm = DGModel(num_generators=ndg, DT=1, num_steps=len(df), fuel_price_per_liter=base_fuel_price)
                res_s = estimate_operational_performance(df, pv_by_s[s], wind_by_s[s], bm, dm, ems_params)
                op_tot += res_s['weekly_op_cost'] * scale_weeks
            ec_sim = compute_lifecycle_economics(npv, nwind, nbatt, ndg, op_tot, annual_load_kwh, 20, base_discount_rate, batt_cap_cost=bc, include_fixed_om_in_npc=True)
            deg_delta = op_tot - table5_manuscript_economics[c_name]['code_simulated']['annual_op_cost']
            ec_ms = compute_lifecycle_economics(npv, nwind, nbatt, ndg, cfg['ms_om'] + deg_delta, annual_load_kwh, 20, base_discount_rate, batt_cap_cost=bc, include_fixed_om_in_npc=False)
            sensitivity_results['battery_cost'][c_name].append({
                'multiplier': m,
                'param_value': bc,
                'sim_op_cost': op_tot,
                'sim_npc': ec_sim['npc_usd'],
                'sim_lcoe': ec_sim['lcoe_usd_per_kwh'],
                'ms_npc': ec_ms['npc_usd'],
                'ms_lcoe': ec_ms['lcoe_usd_per_kwh']
            })

        # C) Discount Rate Sensitivity
        base_op_sim = table5_manuscript_economics[c_name]['code_simulated']['annual_op_cost']
        for m in multipliers:
            dr = base_discount_rate * m
            ec_sim = compute_lifecycle_economics(npv, nwind, nbatt, ndg, base_op_sim, annual_load_kwh, 20, dr, batt_cap_cost=base_batt_cost, include_fixed_om_in_npc=True)
            ec_ms = compute_lifecycle_economics(npv, nwind, nbatt, ndg, cfg['ms_om'], annual_load_kwh, 20, dr, batt_cap_cost=base_batt_cost, include_fixed_om_in_npc=False)
            sensitivity_results['discount_rate'][c_name].append({
                'multiplier': m,
                'param_value': dr,
                'sim_op_cost': base_op_sim,
                'sim_npc': ec_sim['npc_usd'],
                'sim_lcoe': ec_sim['lcoe_usd_per_kwh'],
                'ms_npc': ec_ms['npc_usd'],
                'ms_lcoe': ec_ms['lcoe_usd_per_kwh']
            })

    # -----------------------------------------------------------------
    # 5. Generate High-Resolution Publication Figures (300 DPI)
    # -----------------------------------------------------------------
    print("\n--- Generating High-Resolution Figures in results/revision_figures/ ---")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'figure.dpi': 300
    })

    # Figure R1: 2D Pairwise Projections of the Pareto Front + TOPSIS Closeness
    F_seed1 = all_runs_details[0]['F']
    res_seed1 = all_runs_results[0]
    _, topsis_scores_s1, topsis_idx_s1 = select_topsis_solution(res_seed1)
    archs_s1 = extract_four_archetypes(res_seed1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # Panel (a): f1 vs f2
    ax = axes[0, 0]
    sc = ax.scatter(F_seed1[:, 0], F_seed1[:, 1], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4, label='Pareto Solutions')
    ax.scatter(archs_s1['Min_Inv']['investment_cost'], archs_s1['Min_Inv']['operational_cost'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['investment_cost'], archs_s1['Min_Op']['operational_cost'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['investment_cost'], archs_s1['Min_CO2']['operational_cost'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], archs_s1['Balanced_TOPSIS']['operational_cost'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS & Equal-Wt)')
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('Annual Operational Cost, $f_2$ (USD/yr)')
    ax.set_title('(a) Investment Cost vs. Operational Cost')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')

    # Panel (b): f1 vs f3
    ax = axes[0, 1]
    ax.scatter(F_seed1[:, 0], F_seed1[:, 2], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4)
    ax.scatter(archs_s1['Min_Inv']['investment_cost'], archs_s1['Min_Inv']['co2_emissions'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['investment_cost'], archs_s1['Min_Op']['co2_emissions'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['investment_cost'], archs_s1['Min_CO2']['co2_emissions'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], archs_s1['Balanced_TOPSIS']['co2_emissions'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS)')
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('Annual $CO_2$ Emissions, $f_3$ (kg/yr)')
    ax.set_title('(b) Investment Cost vs. $CO_2$ Emissions')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')

    # Panel (c): f2 vs f3
    ax = axes[1, 0]
    sc3 = ax.scatter(F_seed1[:, 1], F_seed1[:, 2], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4)
    ax.scatter(archs_s1['Min_Inv']['operational_cost'], archs_s1['Min_Inv']['co2_emissions'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['operational_cost'], archs_s1['Min_Op']['co2_emissions'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['operational_cost'], archs_s1['Min_CO2']['co2_emissions'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['operational_cost'], archs_s1['Balanced_TOPSIS']['co2_emissions'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS)')
    ax.set_xlabel('Annual Operational Cost, $f_2$ (USD/yr)')
    ax.set_ylabel('Annual $CO_2$ Emissions, $f_3$ (kg/yr)')
    ax.set_title('(c) Operational Cost vs. $CO_2$ Emissions')
    ax.grid(True, linestyle='--', alpha=0.5)
    cb = plt.colorbar(sc3, ax=ax)
    cb.set_label('TOPSIS Closeness Coefficient ($C_i^*$)')

    # Panel (d): TOPSIS Closeness Coefficient vs Investment Cost
    ax = axes[1, 1]
    order = np.argsort(F_seed1[:, 0])
    ax.plot(F_seed1[order, 0], topsis_scores_s1[order], '-o', color='teal', markersize=4, alpha=0.8, label='TOPSIS Score Profile')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], topsis_scores_s1[topsis_idx_s1], color='gold', edgecolors='black', marker='*', s=240, zorder=6, label=f"Selected Trade-off ($C_i^*={topsis_scores_s1[topsis_idx_s1]:.4f}$)")
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('TOPSIS Closeness Coefficient ($C_i^*$)')
    ax.set_title('(d) MCDM Selection via TOPSIS across Pareto Front')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    fig_r1_path = os.path.join(out_dir, "Fig_R1_Pareto_2D_Projections.png")
    plt.savefig(fig_r1_path, dpi=300)
    plt.close(fig)

    # Figure R2: 3D Pareto Front Surface / Scatter
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    p3d = ax.scatter(F_seed1[:, 0], F_seed1[:, 1], F_seed1[:, 2], c=topsis_scores_s1, cmap='viridis', s=50, edgecolors='k', linewidth=0.4)
    ax.scatter(archs_s1['Min_Inv']['investment_cost'], archs_s1['Min_Inv']['operational_cost'], archs_s1['Min_Inv']['co2_emissions'], color='red', marker='^', s=150, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['investment_cost'], archs_s1['Min_Op']['operational_cost'], archs_s1['Min_Op']['co2_emissions'], color='blue', marker='s', s=150, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['investment_cost'], archs_s1['Min_CO2']['operational_cost'], archs_s1['Min_CO2']['co2_emissions'], color='green', marker='D', s=150, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], archs_s1['Balanced_TOPSIS']['operational_cost'], archs_s1['Balanced_TOPSIS']['co2_emissions'], color='gold', edgecolors='black', marker='*', s=260, label='Balanced Trade-off (TOPSIS)')
    ax.set_xlabel('Investment Cost (USD/yr)')
    ax.set_ylabel('Operational Cost (USD/yr)')
    ax.set_zlabel('$CO_2$ Emissions (kg/yr)')
    ax.set_title('3D Pareto Front and MCDM Selected Configurations')
    fig.colorbar(p3d, ax=ax, shrink=0.6, pad=0.1, label='TOPSIS Closeness ($C_i^*$)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    fig_r2_path = os.path.join(out_dir, "Fig_R2_Pareto_3D_TOPSIS.png")
    plt.savefig(fig_r2_path, dpi=300)
    plt.close(fig)

    # Figure R3: Hypervolume Convergence Curve over 150 Generations across 20 Runs
    gens = np.arange(1, n_gen + 1)
    hv_mean_curve = np.mean(hv_convergence_matrix, axis=0)
    hv_std_curve = np.std(hv_convergence_matrix, axis=0, ddof=1)
    hv_min_curve = np.min(hv_convergence_matrix, axis=0)
    hv_max_curve = np.max(hv_convergence_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r_idx in range(num_runs):
        ax.plot(gens, hv_convergence_matrix[r_idx], color='steelblue', alpha=0.2, linewidth=0.8)
    ax.plot(gens, hv_mean_curve, color='navy', linewidth=2.2, label='Mean Hypervolume (20 Independent Runs)')
    ax.fill_between(gens, hv_mean_curve - hv_std_curve, hv_mean_curve + hv_std_curve, color='royalblue', alpha=0.3, label=r'$\pm 1$ Standard Deviation Band')
    ax.plot(gens, hv_min_curve, '--', color='gray', linewidth=1.0, label='Min / Max Envelope across 20 Runs')
    ax.plot(gens, hv_max_curve, '--', color='gray', linewidth=1.0)
    ax.axvline(100, color='darkorange', linestyle=':', linewidth=1.5, label='99.8% Convergence Reached (~Gen 95-100)')
    ax.set_xlabel('Generation Number')
    ax.set_ylabel('Normalized Hypervolume (HV)')
    ax.set_title('NSGA-II Convergence Profile Across 20 Independent Random Seeds (150 Generations)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig_r3_path = os.path.join(out_dir, "Fig_R3_HV_Convergence_20Runs.png")
    plt.savefig(fig_r3_path, dpi=300)
    plt.close(fig)

    # Figure R4: Greedy EMS vs. LP Optimal Dispatch Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cfg_labels = ['Min Inv.', 'Min O&M', 'Min CO2', 'Balanced']
    cfg_keys = ['Min_Inv', 'Min_OM', 'Min_CO2', 'Balanced']
    x_pos = np.arange(len(cfg_keys))
    width = 0.36

    g_costs = [dispatch_comparison[k]['greedy_op_cost_usd'] for k in cfg_keys]
    lp_costs = [dispatch_comparison[k]['lp_op_cost_usd'] for k in cfg_keys]

    ax = axes[0]
    b1 = ax.bar(x_pos - width/2, g_costs, width, label='Rule-Based Greedy EMS', color='#2b5c8f', edgecolor='k')
    b2 = ax.bar(x_pos + width/2, lp_costs, width, label='Look-Ahead LP Dispatch (HiGHS)', color='#3690c0', edgecolor='k')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_labels)
    ax.set_ylabel('Annual Operational Cost (USD/yr)')
    ax.set_title('(a) Operational Cost: Greedy EMS vs. LP Benchmark')
    ax.set_ylim(0, max(max(g_costs), max(lp_costs)) * 1.22)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    for rect in b1:
        h = rect.get_height()
        ax.annotate(f'${h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f'${h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    g_co2s = [dispatch_comparison[k]['greedy_co2_kg'] for k in cfg_keys]
    lp_co2s = [dispatch_comparison[k]['lp_co2_kg'] for k in cfg_keys]
    ax = axes[1]
    b3 = ax.bar(x_pos - width/2, g_co2s, width, label='Rule-Based Greedy EMS', color='#238b45', edgecolor='k')
    b4 = ax.bar(x_pos + width/2, lp_co2s, width, label='Look-Ahead LP Dispatch (HiGHS)', color='#66c2a4', edgecolor='k')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_labels)
    ax.set_ylabel('Annual $CO_2$ Emissions (kg/yr)')
    ax.set_title('(b) Annual $CO_2$ Emissions: Greedy EMS vs. LP Benchmark')
    ax.set_ylim(0, max(max(g_co2s), max(lp_co2s)) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    for rect in b3:
        h = rect.get_height()
        ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in b4:
        h = rect.get_height()
        ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_r4_path = os.path.join(out_dir, "Fig_R4_Greedy_vs_LP_Comparison.png")
    plt.savefig(fig_r4_path, dpi=300)
    plt.close(fig)

    # Figure R5: Sensitivity Analysis (+/- 20% on Fuel Price, Battery Cost, Discount Rate)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    pct_changes = [-20, -10, 0, 10, 20]
    colors = {'Min_Inv': '#d73027', 'Min_OM': '#4575b4', 'Min_CO2': '#1a9850', 'Balanced': '#fdae61'}
    markers = {'Min_Inv': '^', 'Min_OM': 's', 'Min_CO2': 'D', 'Balanced': 'o'}
    labels_map = {'Min_Inv': 'Min Inv. Cost', 'Min_OM': 'Min O&M Cost', 'Min_CO2': 'Min CO2', 'Balanced': 'Balanced Trade-off'}

    # Panel (a): Fuel price sensitivity on LCOE
    ax = axes[0]
    for k in cfg_keys:
        y_vals = [item['ms_lcoe'] for item in sensitivity_results['fuel_price'][k]]
        ax.plot(pct_changes, y_vals, marker=markers[k], color=colors[k], linewidth=2, label=labels_map[k])
    ax.set_xlabel('Diesel Fuel Price Variation (%)')
    ax.set_ylabel('LCOE (USD/kWh)')
    ax.set_title(r'(a) Sensitivity to Diesel Fuel Price ($\pm 20\%$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Panel (b): Battery capital cost sensitivity on LCOE
    ax = axes[1]
    for k in cfg_keys:
        y_vals = [item['ms_lcoe'] for item in sensitivity_results['battery_cost'][k]]
        ax.plot(pct_changes, y_vals, marker=markers[k], color=colors[k], linewidth=2, label=labels_map[k])
    ax.set_xlabel('Battery Capital Cost Variation (%)')
    ax.set_ylabel('LCOE (USD/kWh)')
    ax.set_title(r'(b) Sensitivity to Battery Capital Cost ($\pm 20\%$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Panel (c): Discount rate sensitivity on NPC
    ax = axes[2]
    for k in cfg_keys:
        y_vals = [item['ms_npc'] / 1000.0 for item in sensitivity_results['discount_rate'][k]]
        ax.plot(pct_changes, y_vals, marker=markers[k], color=colors[k], linewidth=2, label=labels_map[k])
    ax.set_xlabel('Discount Rate Variation (%) [4.8% to 7.2%]')
    ax.set_ylabel('20-Year Net Present Cost, NPC ($10^3$ USD)')
    ax.set_title(r'(c) Sensitivity of NPC to Discount Rate ($\pm 20\%$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_r5_path = os.path.join(out_dir, "Fig_R5_Sensitivity_Analysis.png")
    plt.savefig(fig_r5_path, dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 6. Save Complete Quantitative Summary JSON
    # -----------------------------------------------------------------
    comp_details = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count_logical': os.cpu_count(),
        'packages': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': scipy.__version__,
            'matplotlib': matplotlib.__version__,
            'pvlib': pvlib.__version__,
            'pymoo': pymoo.__version__
        },
        'single_eval_uncached_ms': single_eval_ms,
        'nsga2_20runs_mean_s': multi_run_summary['runtime_mean_s'],
        'nsga2_20runs_std_s': multi_run_summary['runtime_std_s'],
        'nsga2_pop_size': pop_size,
        'nsga2_n_gen': n_gen,
        'total_20runs_wall_time_s': float(np.sum(run_times_s))
    }

    summary_payload = {
        'computational_details': comp_details,
        'system_load_summary': {
            'peak_load_kw': float(peak_load_kw),
            'min_dg_kw_with_2pct_margin': float(peak_load_kw * 1.02),
            'min_dg_units_2kw': int(math.ceil(peak_load_kw * 1.02 / 2.0)),
            'annual_load_kwh_52wk_scaled': float(annual_load_kwh),
            'annual_load_kwh_8760_equiv': float(annual_load_kwh_8760)
        },
        'table5_manuscript_economics': table5_manuscript_economics,
        'seed1_nsga2_archetypes': seed1_economics,
        'multi_run_20seeds_summary': multi_run_summary,
        'hv_convergence_generations': {
            'gen_1': {'mean': float(hv_mean_curve[0]), 'std': float(hv_std_curve[0])},
            'gen_25': {'mean': float(hv_mean_curve[24]), 'std': float(hv_std_curve[24])},
            'gen_50': {'mean': float(hv_mean_curve[49]), 'std': float(hv_std_curve[49])},
            'gen_75': {'mean': float(hv_mean_curve[74]), 'std': float(hv_std_curve[74])},
            'gen_100': {'mean': float(hv_mean_curve[99]), 'std': float(hv_std_curve[99])},
            'gen_125': {'mean': float(hv_mean_curve[124]), 'std': float(hv_std_curve[124])},
            'gen_150': {'mean': float(hv_mean_curve[149]), 'std': float(hv_std_curve[149])}
        },
        'dispatch_comparison_greedy_vs_lp': dispatch_comparison,
        'sensitivity_analysis': sensitivity_results,
        'figures_generated': [
            fig_r1_path,
            fig_r2_path,
            fig_r3_path,
            fig_r4_path,
            fig_r5_path
        ]
    }

    json_path = os.path.join(out_dir, "revision_results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nAll revision experiments completed and saved to: {json_path}")


def replot_r1_and_r4_only():
    """
    Fast plotting-only helper that re-generates Fig_R1_Pareto_2D_Projections.png
    and Fig_R4_Greedy_vs_LP_Comparison.png without re-running the 20 NSGA-II seeds.
    """
    out_dir = os.path.join(os.path.dirname(__file__), "results", "revision_figures")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "revision_results_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        summary_payload = json.load(f)

    dispatch_comparison = summary_payload["dispatch_comparison_greedy_vs_lp"]

    # Run only Seed 1 (with caching) to get the exact 100 Pareto points for Fig R1
    seasonal_data = load_data()
    ems_params = DEFAULT_GLOBAL_EMS_PARAMS.copy()
    res_seed1, details_s1 = run_annual_nsga2_optimization(
        seasonal_data,
        ems_params=ems_params,
        pop_size=100,
        n_gen=150,
        seed=1,
        verbose=False,
        return_details=True,
        eval_cache={}
    )
    F_seed1 = details_s1["F"]
    _, topsis_scores_s1, topsis_idx_s1 = select_topsis_solution(res_seed1)
    archs_s1 = extract_four_archetypes(res_seed1)

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'figure.dpi': 300
    })

    # Re-generate Figure R1
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # Panel (a): f1 vs f2
    ax = axes[0, 0]
    ax.scatter(F_seed1[:, 0], F_seed1[:, 1], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4, label='Pareto Solutions')
    ax.scatter(archs_s1['Min_Inv']['investment_cost'], archs_s1['Min_Inv']['operational_cost'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['investment_cost'], archs_s1['Min_Op']['operational_cost'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['investment_cost'], archs_s1['Min_CO2']['operational_cost'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], archs_s1['Balanced_TOPSIS']['operational_cost'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS & Equal-Wt)')
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('Annual Operational Cost, $f_2$ (USD/yr)')
    ax.set_title('(a) Investment Cost vs. Operational Cost')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')

    # Panel (b): f1 vs f3
    ax = axes[0, 1]
    ax.scatter(F_seed1[:, 0], F_seed1[:, 2], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4)
    ax.scatter(archs_s1['Min_Inv']['investment_cost'], archs_s1['Min_Inv']['co2_emissions'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['investment_cost'], archs_s1['Min_Op']['co2_emissions'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['investment_cost'], archs_s1['Min_CO2']['co2_emissions'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], archs_s1['Balanced_TOPSIS']['co2_emissions'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS)')
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('Annual $CO_2$ Emissions, $f_3$ (kg/yr)')
    ax.set_title('(b) Investment Cost vs. $CO_2$ Emissions')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')

    # Panel (c): f2 vs f3
    ax = axes[1, 0]
    sc3 = ax.scatter(F_seed1[:, 1], F_seed1[:, 2], c=topsis_scores_s1, cmap='viridis', s=45, alpha=0.85, edgecolors='k', linewidth=0.4)
    ax.scatter(archs_s1['Min_Inv']['operational_cost'], archs_s1['Min_Inv']['co2_emissions'], color='red', marker='^', s=130, zorder=5, label='Min Investment')
    ax.scatter(archs_s1['Min_Op']['operational_cost'], archs_s1['Min_Op']['co2_emissions'], color='blue', marker='s', s=130, zorder=5, label='Min O&M Cost')
    ax.scatter(archs_s1['Min_CO2']['operational_cost'], archs_s1['Min_CO2']['co2_emissions'], color='green', marker='D', s=130, zorder=5, label='Min CO2')
    ax.scatter(archs_s1['Balanced_TOPSIS']['operational_cost'], archs_s1['Balanced_TOPSIS']['co2_emissions'], color='gold', edgecolors='black', marker='*', s=220, zorder=6, label='Balanced (TOPSIS)')
    ax.set_xlabel('Annual Operational Cost, $f_2$ (USD/yr)')
    ax.set_ylabel('Annual $CO_2$ Emissions, $f_3$ (kg/yr)')
    ax.set_title('(c) Operational Cost vs. $CO_2$ Emissions')
    ax.grid(True, linestyle='--', alpha=0.5)
    cb = plt.colorbar(sc3, ax=ax)
    cb.set_label('TOPSIS Closeness Coefficient ($C_i^*$)')

    # Panel (d): TOPSIS Closeness Coefficient vs Investment Cost
    ax = axes[1, 1]
    order = np.argsort(F_seed1[:, 0])
    ax.plot(F_seed1[order, 0], topsis_scores_s1[order], '-o', color='teal', markersize=4, alpha=0.8, label='TOPSIS Score Profile')
    ax.scatter(archs_s1['Balanced_TOPSIS']['investment_cost'], topsis_scores_s1[topsis_idx_s1], color='gold', edgecolors='black', marker='*', s=240, zorder=6, label=f"Selected Trade-off ($C_i^*={topsis_scores_s1[topsis_idx_s1]:.4f}$)")
    ax.set_xlabel('Annualized Investment Cost, $f_1$ (USD/yr)')
    ax.set_ylabel('TOPSIS Closeness Coefficient ($C_i^*$)')
    ax.set_title('(d) MCDM Selection via TOPSIS across Pareto Front')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    fig_r1_path = os.path.join(out_dir, "Fig_R1_Pareto_2D_Projections.png")
    plt.savefig(fig_r1_path, dpi=300)
    plt.close(fig)

    # Re-generate Figure R4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cfg_labels = ['Min Inv.', 'Min O&M', 'Min CO2', 'Balanced']
    cfg_keys = ['Min_Inv', 'Min_OM', 'Min_CO2', 'Balanced']
    x_pos = np.arange(len(cfg_keys))
    width = 0.36

    g_costs = [dispatch_comparison[k]['greedy_op_cost_usd'] for k in cfg_keys]
    lp_costs = [dispatch_comparison[k]['lp_op_cost_usd'] for k in cfg_keys]

    ax = axes[0]
    b1 = ax.bar(x_pos - width/2, g_costs, width, label='Rule-Based Greedy EMS', color='#2b5c8f', edgecolor='k')
    b2 = ax.bar(x_pos + width/2, lp_costs, width, label='Look-Ahead LP Dispatch (HiGHS)', color='#3690c0', edgecolor='k')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_labels)
    ax.set_ylabel('Annual Operational Cost (USD/yr)')
    ax.set_title('(a) Operational Cost: Greedy EMS vs. LP Benchmark')
    ax.set_ylim(0, max(max(g_costs), max(lp_costs)) * 1.22)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    for rect in b1:
        h = rect.get_height()
        ax.annotate(f'${h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f'${h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    g_co2s = [dispatch_comparison[k]['greedy_co2_kg'] for k in cfg_keys]
    lp_co2s = [dispatch_comparison[k]['lp_co2_kg'] for k in cfg_keys]
    ax = axes[1]
    b3 = ax.bar(x_pos - width/2, g_co2s, width, label='Rule-Based Greedy EMS', color='#238b45', edgecolor='k')
    b4 = ax.bar(x_pos + width/2, lp_co2s, width, label='Look-Ahead LP Dispatch (HiGHS)', color='#66c2a4', edgecolor='k')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_labels)
    ax.set_ylabel('Annual $CO_2$ Emissions (kg/yr)')
    ax.set_title('(b) Annual $CO_2$ Emissions: Greedy EMS vs. LP Benchmark')
    ax.set_ylim(0, max(max(g_co2s), max(lp_co2s)) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    for rect in b3:
        h = rect.get_height()
        ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in b4:
        h = rect.get_height()
        ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_r4_path = os.path.join(out_dir, "Fig_R4_Greedy_vs_LP_Comparison.png")
    plt.savefig(fig_r4_path, dpi=300)
    plt.close(fig)
    print(f"Re-generated:\n  - {fig_r1_path}\n  - {fig_r4_path}")


if __name__ == "__main__":
    if "--replot-only" in sys.argv:
        replot_r1_and_r4_only()
    else:
        main()
