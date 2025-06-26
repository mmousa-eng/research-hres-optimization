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
    def __init__(self, seasonal_data, ems_params=None):
        self.seasonal_data = seasonal_data
        self.ems_params = ems_params or DEFAULT_GLOBAL_EMS_PARAMS

        # Compute peak load from all seasonal data combined (kW)
        all_loads = np.concatenate([df["Load (kW)"].values for df in seasonal_data.values()])
        self.peak_load_kw = np.max(all_loads)
        self.min_dg_kw = self.peak_load_kw * 1.02  # Add 2% margin

        super().__init__(
            n_var=4,
            n_obj=3,  # investment cost, operational cost, CO2 emissions
            n_constr=1,  # one constraint for minimum DG capacity
            xl=np.array([2.0, 0.0, 2.0, 5.0]),  # lower bounds [pv_kw, wind_kw, batt_kwh, dg_kw]
            xu=np.array([40.0, 100.0, 59.5, 100.0])  # upper bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pv_kw, wind_kw, batt_kwh, dg_kw = x
        pv_w = pv_kw * 1000
        wind_w = wind_kw * 1000
        batt_wh = batt_kwh * 1000
        dg_w = dg_kw * 1000

        num_pv = max(1, math.ceil(pv_w / PV_MODULE_RATING_W))
        num_wind = max(0, math.ceil(wind_w / WIND_TURBINE_RATING_W))
        num_batt = max(1, math.ceil(batt_wh / BATTERY_MODULE_ENERGY_WH))
        num_dg = max(1, math.ceil(dg_w / DG_UNIT_RATING_W))

        dummy = np.array([0])
        pv_model = PVModel(num_modules=num_pv, G=dummy, Ta=dummy, V=dummy)
        wind_model = WindModel(num_turbines=num_wind, V=dummy)
        batt_model = BatteryModel(num_batteries=num_batt, DT=1, num_steps=1)
        dg_model = DGModel(num_generators=num_dg, DT=1, num_steps=1)

        inv_cost = pv_model.compute_total_cost(num_pv) + \
                    wind_model.compute_total_cost(num_wind) + \
                    batt_model.compute_total_cost() + \
                    dg_model.compute_total_cost()

        op_cost, co2 = 0, 0
        for season_name, df in self.seasonal_data.items():
            g = df["G(W/m^2)"].values
            ta = df["Ta(C) at 2m"].values
            v = df["V(m/s) at 10m"].values

            pv_model = PVModel(num_modules=num_pv, G=g, Ta=ta, V=v)
            wind_model = WindModel(num_turbines=num_wind, V=v)
            batt_model = BatteryModel(num_batteries=num_batt, DT=1, num_steps=len(df))
            dg_model = DGModel(num_generators=num_dg, DT=1, num_steps=len(df))

            pv_power = pv_model.power_output()
            wind_power = wind_model.power_output()

            result = estimate_operational_performance(
                season_df=df,
                pv_power_available_w=pv_power,
                wind_power_available_w=wind_power,
                battery_model=batt_model,
                dg_model=dg_model,
                global_params=self.ems_params
            )

            # Scale weekly cost and emissions to annual assuming 52 weeks in a year
            scale = 52 / len(self.seasonal_data)
            op_cost += result['weekly_op_cost'] * scale
            co2 += result['weekly_co2'] * scale

        # Constraint: DG capacity must be at least min_dg_kw
        g_dg = self.min_dg_kw - dg_kw  # <= 0 means feasible
        out["G"] = [g_dg]

        out["F"] = [inv_cost, op_cost, co2]

class CapturePareto(Callback):
    def __init__(self):
        super().__init__()
        self.data = []

    def notify(self, algorithm):
        self.data = algorithm.opt.get("F")

def run_annual_nsga2_optimization(seasonal_data, ems_params):
    problem = HRESProblem(seasonal_data)

    algorithm = NSGA2(pop_size=60)
    termination = get_termination("n_gen", 100)
    capture = CapturePareto()

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        callback=capture,
        save_history=True,
        verbose=True
    )

    pareto_front = result.F
    pareto_solutions = result.X

    results = []
    for i in range(len(pareto_solutions)):
        x = pareto_solutions[i]
        f = pareto_front[i]

        results.append({
            'pv_kw': x[0],
            'wind_kw': x[1],
            'battery_kwh': x[2],
            'dg_kw': x[3],
            'investment_cost': f[0],
            'operational_cost': f[1],
            'co2_emissions': f[2]
        })

    return results
