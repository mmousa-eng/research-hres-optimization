import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from ems import run_ems
from models.load_data import load_data

bounds = {'PV': (2, 29), 'Wind': (1, 2), 'Battery': (2, 74), 'DG': (1, 15)}

def run_ga_for_season(season_df):
    """
    Run Genetic Algorithm for one season.
    Returns best configuration (PV, Wind, Battery, DG) and objective value.
    """

    def fitness_function(X):
        num_pv = int(X[0])
        num_wind = int(X[1])
        num_battery = int(X[2])
        num_dg = int(X[3])

        result = run_ems(season_df, num_pv, num_wind, num_battery, num_dg)

        total_annualized_cost = result['total_operating_cost']
        co2_emissions = result['total_dg_emissions']

        return total_annualized_cost + 0.001 * co2_emissions

    varbound = np.array([[2, 29], [1, 2], [2, 74], [1, 15]])
    algorithm_param = {
        'max_num_iteration': 20,
        'population_size': 10,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }

    model = ga(
        function=fitness_function,
        dimension=4,
        variable_type='int',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param
    )

    model.run()

    best = model.output_dict['variable']
    score = model.output_dict['function']

    print(f"[DEBUG] Best solution for season: PV={int(best[0])}, Wind={int(best[1])}, Batt={int(best[2])}, DG={int(best[3])}, Obj={score:.2f}")

    return {
        'best_pv': int(best[0]),
        'best_wind': int(best[1]),
        'best_battery': int(best[2]),
        'best_dg': int(best[3]),
        'objective_value': score
    }
