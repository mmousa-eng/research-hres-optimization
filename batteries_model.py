class BatteryModel:
    def __init__(self, num_batteries, DT, num_steps, energy_capacity=1190, max_discharge=1190,
                 max_charge=1190, soc_min=0.4, soc_max=0.9, discharge_factor=1.165,
                 capital_cost=2000, lifetime_cycles=1500, converter_efficiency=0.95,
                 maintenance_cost_pct=0.05, lifetime_years=25):
        
        self.DT = DT  # Time step in hours
        self.num_steps = num_steps
        self.lifetime_years = lifetime_years
        self.num_batteries = num_batteries
        self.energy_capacity = energy_capacity  # Wh per battery
        self.max_discharge = max_discharge  # W per battery
        self.max_charge = max_charge  # W per battery
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.discharge_factor = discharge_factor
        self.capital_cost = capital_cost
        self.lifetime_cycles = lifetime_cycles
        self.converter_efficiency = converter_efficiency
        self.maintenance_cost_pct = maintenance_cost_pct

        # Initialize state of charge to midpoint (or user can set it)
        self.current_soc = (self.soc_min + self.soc_max) / 2

    def charge_battery(self, charge_power, duration_hours):
        """
        Charge battery given power and duration in hours.
        """
        energy_charged_Wh = charge_power * duration_hours * self.converter_efficiency
        delta_soc = energy_charged_Wh / (self.num_batteries * self.energy_capacity)
        self.current_soc = min(self.current_soc + delta_soc, self.soc_max)
        return self.current_soc

    def discharge_battery(self, discharge_power, duration_hours):
        energy_discharged_Wh = discharge_power * duration_hours / self.converter_efficiency
        delta_soc = energy_discharged_Wh / (self.num_batteries * self.energy_capacity)
        self.current_soc = max(self.current_soc - delta_soc, self.soc_min)
        return self.current_soc


    def compute_degradation(self, charge_power, discharge_power):
        """
        Compute degradation impact based on charge and discharge powers at current time step.
        """
        degradation = ((0.5 * self.DT) / (self.num_batteries * self.energy_capacity)) * \
                      (charge_power + (discharge_power / self.discharge_factor))
        return degradation

    def compute_total_cost(self):
        """
        Calculate annualized cost = capital cost / lifetime + annual maintenance.
        """
        investment_cost_per_year = (self.num_batteries * self.capital_cost) / self.lifetime_years
        maintenance_cost_per_year = self.num_batteries * self.maintenance_cost_pct * self.capital_cost
        total_annualized_cost = investment_cost_per_year + maintenance_cost_per_year
        return total_annualized_cost
