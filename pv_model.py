import pvlib

class PVModel:
    """
    Models the photovoltaic system performance, cost, and power generation using CEC single-diode model.

    Attributes:
        capital_cost (float): Investment cost per module (USD).
        maintenance_cost_per_year (float): Annual maintenance cost per module (USD).
        lifetime (int): System lifetime in years.
    """
    def __init__(self,
                 num_modules,
                 G, Ta, V, # From the data file
                 pv_effeciency=0.95,
                 capital_cost=7000,
                 maintenance_cost_per_year=20,
                 lifetime=25):
        self.capital_cost = capital_cost
        self.pv_effeciency = pv_effeciency
        self.maintenance_cost_per_year = maintenance_cost_per_year
        self.lifetime = lifetime
        self.num_modules = num_modules
        self.G = G
        self.Ta = Ta
        self.V = V
        # Specs for SolarWorld SW 240 Poly (from CEC database)
        self.alpha_sc = 0.006668
        self.a_ref = 1.603731
        self.I_L_ref = 8.441907
        self.I_o_ref = 1e-10
        self.R_s = 0.292428
        self.R_sh_ref = 1293.893799
        self.Adjust = 9.068874
        self.N_s = 60

    def power_output(self):
        """
        Calculate total PV power output (W) using single-diode model,
        based on irradiance G (W/m²), ambient temperature Ta (C), and wind speed V (m/s).
        """
        T_cell = pvlib.temperature.sapm_cell(self.Ta, self.V, self.G, a=0.04, b=-0.002, deltaT=3)

        IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_cec(
            effective_irradiance=self.G,
            temp_cell=T_cell,
            alpha_sc=self.alpha_sc,
            a_ref=self.a_ref,
            I_L_ref=self.I_L_ref,
            I_o_ref=self.I_o_ref,
            R_sh_ref=self.R_sh_ref,
            R_s=self.R_s,
            Adjust=self.Adjust
        )

        singlediode_output = pvlib.pvsystem.singlediode(
            photocurrent=IL,
            saturation_current=I0,
            resistance_series=Rs,
            resistance_shunt=Rsh,
            nNsVth=nNsVth
        )

        power_per_module = singlediode_output['p_mp']
        return self.num_modules * power_per_module

    def compute_total_cost(self, num_modules):
        """Annualized (investment + maintenance) (USD)."""
        investment_cost = num_modules * self.capital_cost / self.lifetime
        maintenance_cost = num_modules * self.maintenance_cost_per_year 
        total_annualized_cost = investment_cost + maintenance_cost
        return total_annualized_cost
    