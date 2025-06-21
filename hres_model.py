import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from models.pv_model import PVModel
from models.wind_model import WindModel
from models.dg_model import DieselGeneratorModel
from models.batteries_model import BatteryModel
from models.load_data import load_data
from models.ems_model import ems_optimizer  # EMS module

class HybridRenewableEnergySystem:
    """
    Integrated Hybrid Renewable Energy System (HRES), managing PV, wind, diesel generators, and batteries.
    Now includes EMS optimization.
    """

    def __init__(self, num_modules=14, num_wind=1, num_dg=14, num_batteries=64, data_path="data/Accepted_Week_Data.xlsx"):
        """Initialize system components and load seasonal data."""
        self.pv = PVModel(num_modules=num_modules)
        self.wind = WindModel(num_turbines=num_wind)
        self.dg = DieselGeneratorModel(num_generators=num_dg)
        self.battery = BatteryModel(num_batteries=num_batteries)
        self.seasonal_data = load_data(data_path)  # Load seasonal data

    def compute_system_performance(self, season):
        """
        Computes total energy production, costs, and emissions for a given season.
        Now incorporates EMS optimization.
        """

        if season not in self.seasonal_data:
            print(f"\n❌ Error: No data available for '{season}'.")
            return None

        df = self.seasonal_data[season]

        # Extract load & renewable generation
        load_profile = df["Load (kW)"].tolist()
        pv_output = df["G(W/m^2)"].tolist()
        wind_output = df["V(m/s) at 10m"].tolist()

        # Run EMS optimization
        optimized_dispatch = ems_optimizer(
        self.battery.num_batteries, self.dg.num_generators, self.pv.num_modules, self.wind.num_turbines,
        df, self.battery.energy_capacity, self.dg.efficiency, self.battery.discharge_factor, 1, 0.5, 0.3, 0.2
        )

        # Compute optimized dispatch results
        total_diesel_power = sum([entry["Diesel Generator Power (kWh)"] for entry in optimized_dispatch])
        total_fuel_consumption = self.dg.compute_fuel_consumption(total_diesel_power)
        total_co2_emissions = self.dg.compute_co2_emissions(total_diesel_power)
        total_battery_degradation = sum([entry["Battery SoC"] for entry in optimized_dispatch]) / len(optimized_dispatch)

        return {
            "Total PV Power (kWh)": sum(pv_output),
            "Total Wind Power (kWh)": sum(wind_output) / 1000,  # Convert to MWh
            "Total DG Power (kWh)": total_diesel_power,
            "Total Fuel Consumption (L)": total_fuel_consumption,
            "Total CO2 Emissions (kg)": total_co2_emissions,
            "Battery Degradation": total_battery_degradation,
            "Total System Cost (USD)": sum([
                self.pv.compute_total_cost(),
                self.wind.compute_total_cost(),
                self.dg.compute_total_cost(),
                self.battery.compute_total_cost()
            ])
        }
