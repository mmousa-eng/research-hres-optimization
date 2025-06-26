import numpy as np

class WindModel:  # Changed WMTodel to WindModel here
    def __init__(self, num_turbines, V,
                 rated_power=20000,  # W
                 cut_in=2.75, rated=7.5, cut_out=20,
                 hub_height=36, ref_height=10, alpha=0.25,
                 capital_cost=38745, turbine_efficiency=0.95, lifetime=25):

        # Turbine performance
        self.V = V  # wind_speed_ref
        self.num_turbines = num_turbines
        self.rated_power = rated_power
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.turbine_efficiency = turbine_efficiency

        # Wind height adjustment
        self.hub_height = hub_height
        self.ref_height = ref_height
        self.alpha = alpha

        # Costs
        self.capital_cost = capital_cost
        self.lifetime = lifetime

    def power_output(self):
        """
        Compute total power output of all wind turbines (W) based on wind speed at hub height (hourly).
        """
        wind_speed_hub = self.V * (self.hub_height / self.ref_height) ** self.alpha

        # Piecewise wind turbine power curve
        power_per_turbine = np.where(
            (wind_speed_hub < self.cut_in) | (wind_speed_hub >= self.cut_out),
            0,
            np.where(
                wind_speed_hub < self.rated,
                self.rated_power * ((wind_speed_hub - self.cut_in) / (self.rated - self.cut_in)) ** 3,
                self.rated_power
            )
        )

        return self.num_turbines * power_per_turbine * self.turbine_efficiency

    def compute_total_cost(self, num_turbines):
        """Annualized investment + maintenance (USD/year)."""
        investment_cost = self.num_turbines * self.capital_cost / self.lifetime
        maintenance_cost = self.num_turbines * self.capital_cost * 0.03
        total_annualized_cost = investment_cost + maintenance_cost
        return num_turbines * total_annualized_cost