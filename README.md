# 🔍 Research: Multi-Objective Optimization of Hybrid Renewable Energy System (HRES) Sizing Using NSGA-II 
This repository presents the simulation and optimization of a standalone Hybrid Renewable Energy System (HRES) designed to supply power to a rural community in Udhruh, southern Jordan.

Technologies: Python • NumPy • Pandas • PVLib • AI Optimization (NSGA-II) • HRES

## 📈 Objective
Sizing optimization of HRES components: PV, Wind, Battery, and Diesel Generator (DG)

## ⚙️ Optimization Approach
Multi-objective algorithm: NSGA-II

EMS: Greedy logic (PV → Wind → Battery → DG → Load Shedding)

## 📊 Sample Output
| Objective             | PV (N) | Wind (N) | Battery (N) | DG (N) | Investment Cost ($) | Operational Cost ($/yr) | CO₂ Emissions (kg/yr) | DG Contribution (%) |
|-----------------------|--------|----------|-------------|--------|---------------------|-------------------------|-----------------------|---------------------|
| Minimum Investment    | 3      | 1        | 2           | 15     | 4,883               | 18,565                  | 28,569                | 98.23               |
| Minimum Operational   | 39     | 1        | 2           | 15     | 15,983              | 18,145                  | 26,681                | 78.13               |
| Minimum CO₂ Emissions | 40     | 4        | 50          | 15     | 89,715              | 58,588                  | 15,025                | 4.23                |
| Best Trade-off        | 3      | 1        | 6           | 16     | 13,980              | 24,909                  | 23,607                | 32.97               |
