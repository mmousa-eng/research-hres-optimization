class DGModel:
    """
    Diesel Generator model for performance, cost, degradation, and CO2 emissions.

    Args:
        num_generators (int): Number of generators.
        rated_power (float): Rated power per generator (kW).
        efficiency (float): Generator efficiency (fraction).
        fuel_price_per_liter (float): Diesel fuel cost ($/L).
        fuel_consumption_factor (float): Fuel consumption per kWh (L/kWh).
        base_load_consumption (float): Base fuel consumption (L/kWh).
        capital_cost (float): Capital cost per generator ($).
        maintenance_cost_per_hour (float): Maintenance cost per running hour ($).
        lifetime (float): Lifetime in years.
        co2_emission_factor (float): CO2 emissions per liter of fuel (kg/L).
        diesel_cfactor (float): Diesel fuel conversion factor.
    """

    def __init__(self, num_generators, DT, num_steps,
                 rated_power=2000, efficiency=0.95, fuel_price_per_liter=1.06,
                 fuel_consumption_factor=0.000246, base_load_consumption=0.00008145,
                 capital_cost=1514, maintenance_cost_per_hour=0.17,
                 lifetime=25, co2_emission_factor=2.6, diesel_cfactor=0.8):
        self.num_generators = num_generators
        self.DT = DT
        self.num_steps = num_steps
        self.rated_power = rated_power
        self.efficiency = efficiency
        self.fuel_price_per_liter = fuel_price_per_liter
        self.fuel_consumption_factor = fuel_consumption_factor
        self.base_load_consumption = base_load_consumption
        self.capital_cost = capital_cost
        self.maintenance_cost_per_hour = maintenance_cost_per_hour
        self.lifetime = lifetime
        self.co2_emission_factor = co2_emission_factor
        self.diesel_cfactor = diesel_cfactor

    def power_output(self, load_demand):
        """
        Calculate power output, fuel consumption, cost, and emissions.

        Args:
            load_demand (float): Load demand in kW.

        Returns:
            dict: Contains power_output_kw, fuel_consumption_l,
                  operating_cost_usd, co2_emissions_kg
        """
        power_output_total = load_demand / self.efficiency
        fuel_consumption = self.num_generators * (
            self.fuel_consumption_factor * power_output_total +
            self.base_load_consumption * self.rated_power
        ) * self.DT
        fuel_cost = fuel_consumption * self.fuel_price_per_liter
        co2_emissions = fuel_consumption * self.co2_emission_factor * self.diesel_cfactor
        operating_cost = fuel_cost + self.num_generators * self.maintenance_cost_per_hour * self.DT

        return {
            'num_generators': self.num_generators,
            'power_output_kw': power_output_total,
            'fuel_consumption_l': fuel_consumption,
            'operating_cost_usd': operating_cost,
            'co2_emissions_kg': co2_emissions
        }

    def compute_total_annualized_cost(self):
        """
        Compute annualized capital + maintenance cost.

        Returns:
            float: total annualized cost in USD
        """
        investment_cost = (self.num_generators * self.capital_cost) / self.lifetime
        duration_hours = self.DT * self.num_steps
        maintenance_cost = self.num_generators * self.maintenance_cost_per_hour * duration_hours
        total_annualized_cost = investment_cost + maintenance_cost
        return total_annualized_cost
