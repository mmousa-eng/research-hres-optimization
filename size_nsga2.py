import numpy as np
import pandas as pd
import math
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from ems import estimate_operational_performance

from models.pv_model import PVModel
from models.wind_model import WindModel
from models.batteries_model import BatteryModel
from models.dg_model import DGModel

# Component ratings
PV_MODULE_RATING_W = 1000
WIND_TURBINE_RATING_W = 20000
BATTERY_MODULE_ENERGY_WH = 1190
BATTERY_MODULE_POWER_W = 1190
DG_UNIT_RATING_W = 2000

# EMS constants
DEFAULT_GLOBAL_EMS_PARAMS = {
    'DT': 1,
    'cost_load_shed_per_wh': 5.6e-3,
    'cost_curtailment_per_wh': 0.1e-3,
    'system_eff_pv': 0.90,
    'system_eff_wind': 0.90,
}

# Cost assumptions
MAX_EXPECTED_INV_COST = 200000


class HRESProblem(ElementwiseProblem):
    def __init__(self, seasonal_data, ems_params=None, eval_cache=None):
        self.seasonal_data = seasonal_data
        self.ems_params = ems_params or DEFAULT_GLOBAL_EMS_PARAMS
        self._eval_cache = eval_cache if eval_cache is not None else {}

        # Compute peak load from all seasonal data combined (kW)
        all_loads = np.concatenate([df["Load (kW)"].values for df in seasonal_data.values()])
        self.peak_load_kw = float(np.max(all_loads))
        self.min_dg_kw = self.peak_load_kw * 1.02  # Add 2% safety margin (Comment 6)
        self.min_dg_units_for_peak = math.ceil((self.peak_load_kw * 1000.0) / DG_UNIT_RATING_W)

        # Pre-compute unit PV (1 module) and unit Wind (1 turbine) power curves and load arrays per season
        self.unit_pv_power = {}
        self.unit_wind_power = {}
        self.load_demand_w = {}
        for season_name, df in self.seasonal_data.items():
            g = df["G(W/m^2)"].values
            ta = df["Ta(C) at 2m"].values
            v = df["V(m/s) at 10m"].values
            self.unit_pv_power[season_name] = np.asarray(
                PVModel(num_modules=1, G=g, Ta=ta, V=v).power_output(), dtype=float
            )
            self.unit_wind_power[season_name] = np.asarray(
                WindModel(num_turbines=1, V=v).power_output(), dtype=float
            )
            self.load_demand_w[season_name] = df["Load (kW)"].values * 1000.0

        super().__init__(
            n_var=4,
            n_obj=3,  # investment cost, operational cost, CO2 emissions
            n_constr=1,  # one constraint for minimum DG capacity
            xl=np.array([2.0, 0.0, 2.0, 5.0]),  # lower bounds [pv_kw, wind_kw, batt_kwh, dg_kw]
            xu=np.array([40.0, 100.0, 59.5, 100.0])  # upper bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pv_kw, wind_kw, batt_kwh, dg_kw = x
        pv_w = pv_kw * 1000.0
        wind_w = wind_kw * 1000.0
        batt_wh = batt_kwh * 1000.0
        dg_w = dg_kw * 1000.0

        num_pv = max(1, math.ceil(pv_w / PV_MODULE_RATING_W))
        num_wind = max(0, math.ceil(wind_w / WIND_TURBINE_RATING_W))
        num_batt = max(1, math.ceil(batt_wh / BATTERY_MODULE_ENERGY_WH))
        num_dg = max(1, math.ceil(dg_w / DG_UNIT_RATING_W))

        dummy = np.array([0])
        pv_model = PVModel(num_modules=num_pv, G=dummy, Ta=dummy, V=dummy)
        wind_model = WindModel(num_turbines=num_wind, V=dummy)
        batt_model = BatteryModel(num_batteries=num_batt, DT=1, num_steps=1)
        dg_model = DGModel(num_generators=num_dg, DT=1, num_steps=1)

        inv_cost = (
            pv_model.compute_total_cost(num_pv) +
            wind_model.compute_total_cost(num_wind) +
            batt_model.compute_total_cost() +
            dg_model.compute_total_cost()
        )

        # Since peak load is covered by min_dg_units_for_peak (15 units), any num_dg >= 15 produces
        # identical hourly dispatch. Caching by effective DG count accelerates multi-seed runs.
        op_key = (num_pv, num_wind, num_batt, min(num_dg, self.min_dg_units_for_peak))
        if op_key in self._eval_cache:
            op_cost, co2 = self._eval_cache[op_key]
        else:
            op_cost, co2 = 0.0, 0.0
            scale = 52.0 / len(self.seasonal_data)
            for season_name, df in self.seasonal_data.items():
                batt_sim = BatteryModel(num_batteries=num_batt, DT=1, num_steps=len(df))
                dg_sim = DGModel(num_generators=num_dg, DT=1, num_steps=len(df))

                pv_power = num_pv * self.unit_pv_power[season_name]
                wind_power = num_wind * self.unit_wind_power[season_name]

                result = estimate_operational_performance(
                    season_df=df,
                    pv_power_available_w=pv_power,
                    wind_power_available_w=wind_power,
                    battery_model=batt_sim,
                    dg_model=dg_sim,
                    global_params=self.ems_params,
                    load_demand_w=self.load_demand_w[season_name]
                )

                op_cost += result['weekly_op_cost'] * scale
                co2 += result['weekly_co2'] * scale
            self._eval_cache[op_key] = (op_cost, co2)

        # Constraint: DG capacity must be at least min_dg_kw
        g_dg = self.min_dg_kw - dg_kw  # <= 0 means feasible
        out["G"] = [g_dg]
        out["F"] = [inv_cost, op_cost, co2]


class CapturePareto(Callback):
    def __init__(self):
        super().__init__()
        self.data = []
        self.history_F = []

    def notify(self, algorithm):
        current_f = algorithm.opt.get("F")
        self.data = current_f
        if current_f is not None and len(current_f) > 0:
            self.history_F.append(np.array(current_f, copy=True))


def run_annual_nsga2_optimization(
    seasonal_data,
    ems_params=None,
    pop_size=60,
    n_gen=150,
    seed=1,
    verbose=False,
    return_details=False,
    eval_cache=None
):
    problem = HRESProblem(seasonal_data, ems_params=ems_params, eval_cache=eval_cache)

    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)
    capture = CapturePareto()

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        callback=capture,
        save_history=False,
        verbose=verbose
    )

    pareto_front = result.F
    pareto_solutions = result.X

    results = []
    for i in range(len(pareto_solutions)):
        x = pareto_solutions[i]
        f = pareto_front[i]

        results.append({
            'pv_kw': float(x[0]),
            'wind_kw': float(x[1]),
            'battery_kwh': float(x[2]),
            'dg_kw': float(x[3]),
            'num_pv': max(1, math.ceil(x[0] * 1000.0 / PV_MODULE_RATING_W)),
            'num_wind': max(0, math.ceil(x[1] * 1000.0 / WIND_TURBINE_RATING_W)),
            'num_batt': max(1, math.ceil(x[2] * 1000.0 / BATTERY_MODULE_ENERGY_WH)),
            'num_dg': max(1, math.ceil(x[3] * 1000.0 / DG_UNIT_RATING_W)),
            'investment_cost': float(f[0]),
            'operational_cost': float(f[1]),
            'co2_emissions': float(f[2])
        })

    if return_details:
        return results, {
            'F': pareto_front,
            'X': pareto_solutions,
            'history_F': capture.history_F,
            'problem': problem
        }
    return results
