## 🔍 Research: Multi-Objective Optimization of a Hybrid Renewable Energy System
This repository presents the full simulation and optimization of a standalone Hybrid Renewable Energy System (HRES) designed to supply power to a rural community in Udhruh, southern Jordan.

Technologies: Python • NSGA-II • PVLib • NumPy • Pandas

## 📈 Objectives
🔧 Design optimization of the HRES components: PV, Wind, Battery, and Diesel Generator (DG)

## ⚙️ Optimization Approach
Multi-objective algorithm: NSGA-II

EMS: Greedy logic (PV → Wind → Battery → DG → Load Shedding)

## 📁 Project Structure
bash
Copy
Edit
research-hres-optimization/
│
├── main_ga.py              # Run optimization using Genetic Algorithm
├── main_nsga2.py           # Run optimization using NSGA-II
├── size_ga.py              # GA logic
├── size_nsga2.py           # NSGA-II logic
├── ems.py                  # Rule-based energy dispatch
├── utils/
│   └── load_data.py        # Loads Excel weather/load input data
├── models/
│   ├── pv_model.py         # PV system simulation
│   ├── wind_model.py       # Wind turbine simulation
│   ├── batteries_model.py  # Battery system simulation
│   └── dg_model.py         # Diesel generator model
├── data/
│   └── Accepted_Week_Data.xlsx  # Input data (hourly for four seasonal weeks)
└── .gitignore

## 🚀 Running the Optimization

Run the desired optimization script, e.g.:

```bash
python main_nsga2.py

📊 Sample Output (Summary)
Objective	PV (N)	Wind (N)	Battery (N)	DG (N)	CAPEX ($)	OPEX ($/yr)	CO₂ (kg/yr)
Min Investment	3	1	2	15	4,883	18,565	28,569
Min OPEX	39	1	2	15	15,983	18,144	26,681
Min Emissions	40	4	50	15	89,714	58,587	15,025
Best Trade-Off	3	1	6	16	13,980	24,908	23,607
