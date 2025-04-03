# Life, Death, & Aliens Simulation

This repository contains a Python simulation that models an unconventional survival scenario where a community lives high in a tree canopy and faces monthly attrition. After 50 years (600 months), an alien spaceship arrives to rescue any remaining survivors. The simulation is designed to answer two key questions:

1. **Functional Relationship:** What is the relationship between the initial population size and the probability of survival?
2. **Minimum Population:** What is the minimum initial population required to have at least a 99% chance that one or more individuals survive until the alien rescue?

---

## Problem Description

Imagine a community with the following characteristics:

- **Attrition:** Each month, one individual is randomly removed from the population.
- **Reproduction:**
  - Gestation period is 9 months.
  - Only males and females mate.
  - No multiple births occur.
  - Mating with one’s ancestors is not allowed (with a configurable ancestry limit).
  - Individuals become reproductively viable at age 18 (i.e., 216 months).
  - Newborns are equally likely to be male or female.
- **Initial Conditions:** All starting individuals are at least 18 years old.
- **Rescue Event:** An alien spaceship will arrive after 50 years, rescuing any survivors.

---

## Key Components

### Simulation Mechanics

- **Person Class:**  
  Represents an individual with attributes such as gender, birth month, and references to parents. It includes an `age()` method to calculate the individual’s age in months.

- **Ancestry Checks:**  
  Functions `are_related()` and `related_within()` ensure that individuals do not mate with their ancestors within a specified generational limit.

- **Simulator Class:**  
  Handles the simulation by:
  - **Initialization:** Creating an initial population (all aged 18+).
  - **Process Births:** Checking for due pregnancies (after 9 months of gestation) and adding new individuals.
  - **Process Mating:** Enabling reproduction for eligible females by pairing them with non-related, reproductive-age males.
  - **Process Attrition:** Removing one random individual each month.
  - **Data Tracking:** Recording population history and births for analysis.

### Data Analysis

- **Survival Probability:**  
  For various initial population sizes, the simulation calculates the probability of having survivors after 50 years over multiple episodes.

- **Logistic Curve Fitting:**  
  A logistic function is fitted to the survival data to understand the functional relationship between initial population size and survival probability.

- **Statistical Analysis:**  
  A Cox Proportional Hazards model is used to evaluate the impact of initial population size and ancestral breeding restrictions on survival outcomes.

---

## Dependencies

The simulation requires the following Python libraries:

- **Standard Libraries:** `os`, `csv`, `random`
- **Third-party Libraries:**
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `statsmodels`
  - `lifelines` (install with `pip install lifelines`)
  - `scipy`

Ensure you have Python 3 installed, and then install the required packages:

```bash
pip install matplotlib numpy pandas statsmodels lifelines scipy
```

---

## Usage

To run the simulation, execute:

```bash
python PopulationSimulator.py
```

If you want to use preloaded data, save the survival_data.csv file in the same directory as the code

### What Happens When You Run the Simulation?

- **Simulation Duration:**  
  The simulation runs for 600 months (50 years).

- **Multiple Episodes:**  
  It performs multiple simulation runs (episodes) for varying initial population sizes.

- **Data Logging:**  
  Results are saved in a CSV file (`survival_data.csv`), allowing the simulation to resume from previous runs.

- **Output:**
  - A plot showing survival probability vs. initial population size along with the fitted logistic curve.
  - Console output that includes:
    - Fitted logistic curve parameters.
    - The minimum initial population required to have at least a 99% chance of survival.
    - Statistical significance results from the Cox Proportional Hazards model.

---

## Code Structure

- **Person Class:**  
  Represents each individual with attributes for gender, birth month, and parental links.

- **Ancestry Functions:**  
  `are_related()` and `related_within()` check for prohibited matings based on ancestry.

- **Simulator Class:**  
  Manages the simulation lifecycle by processing births, mating, and monthly attrition while tracking population statistics.

- **Main Execution Block:**  
  Runs the simulation over a range of initial population sizes and episodes, logs data to a CSV file, computes survival probabilities, fits a logistic model, and performs survival analysis using a Cox Proportional Hazards model.

---

## Contributing

Contributions to improve or extend the simulation are welcome. Feel free to fork the repository.

---

## Acknowledgments

Special thanks to Samira for inspiring the problem that forms the basis of this simulation.

---
