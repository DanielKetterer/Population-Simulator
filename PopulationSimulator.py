"""
Life, Death, & Aliens:

(This problem was stolen wholeheartedly from my coworker Samira, I hope it still resembles the one she mentioned.)

Imagine a community of people living in a tree canopy.
Each month, the population decreases by one individual due to some unspecified, random selection form of attrition.
In 50 years, an alien spaceship will arrive to rescue any remaining survivors.


We assume the following:
The gestation period is 9 months.
Only males and females can mate.
No multiple births occur.
No breeding with one's ancestors at any level.
Individuals become reproductively viable at age 18.
Equal odds of birthing a male or female child.
All starting individuals are at least 18.

Questions:

1. What functional relationship, if any, does initial population and survival probability have?

2. What is the minimum initial population required to have at least a 99% chance of there being at least one survivor remaining after 50 years?

"""
import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter  
from scipy.optimize import curve_fit

# Define the logistic function with upper limit 1
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

class Person:
    id_counter = 0
    
    def __init__(self, gender, birth_month, father=None, mother=None):
        self.id = Person.id_counter
        Person.id_counter += 1
        self.gender = gender            # 'M' or 'F'
        self.birth_month = birth_month  # month when born
        self.father = father            # reference to father's Person object (if known)
        self.mother = mother            # reference to mother's Person object (if known)
        self.is_pregnant = False        # only applicable for females

    def age(self, current_month):
        """Return the age in months given the current simulation month."""
        return current_month - self.birth_month

    def __repr__(self):
        return f"Person(id={self.id}, gender={self.gender}, birth_month={self.birth_month})"

def are_related(person1, person2, depth):
    """
    Checks if person1 is an ancestor of person2 (or vice versa) within the given depth.
    For depth == 1, only a direct parent is considered; depth == 2 includes grandparents, etc.
    """
    if depth <= 0:
        return False
    if person2.mother == person1 or person2.father == person1:
        return True
    if person2.mother is not None:
        if are_related(person1, person2.mother, depth - 1):
            return True
    if person2.father is not None:
        if are_related(person1, person2.father, depth - 1):
            return True
    return False

def related_within(person1, person2, ancestry_limit):
    """
    Returns True if person1 and person2 are related within the specified number of generations.
    We check both directions (i.e. whether person1 is an ancestor of person2 and vice versa).
    """
    return are_related(person1, person2, ancestry_limit) or are_related(person2, person1, ancestry_limit)

class Simulator:
    def __init__(self, initial_population_size, simulation_months, ancestry_limit, verbose=True):
        self.current_month = 0
        self.population = []
        self.pregnancies = []  # Each pregnancy is a tuple: (mother, father, start_month)
        self.simulation_months = simulation_months
        self.ancestry_limit = ancestry_limit  # Maximum allowed ancestry level for mating
        self.verbose = verbose
        
        # History trackers for plotting per simulation
        self.population_history = []
        self.births_history = []
        
        # Initialize population with random genders.
        # Setting birth_month to -216 ensures each individual is at least 18 (216 months old) at month 0.
        for _ in range(initial_population_size):
            gender = random.choice(['M', 'F'])
            person = Person(gender, birth_month=-216)
            self.population.append(person)
    
    def run(self):
        # Record the initial state
        self.population_history.append(len(self.population))
        self.births_history.append(0)  # No births at month 0
        
        for month in range(1, self.simulation_months + 1):
            self.current_month = month
            births_this_month = 0
            
            if self.verbose:
                print(f"\nMonth {month} starting with population: {len(self.population)}")
            
            births_this_month = self.process_births()
            self.process_mating()
            self.process_attrition()
            
            self.population_history.append(len(self.population))
            self.births_history.append(births_this_month)
            
            if self.verbose:
                print(f"Month {month} ended with population: {len(self.population)}")
    
    def process_births(self):
        # Check if any pregnancy is due (gestation period = 9 months)
        births_due = [preg for preg in self.pregnancies if self.current_month - preg[2] >= 9]
        births_count = 0
        for pregnancy in births_due:
            mother, father, start_month = pregnancy
            gender = random.choice(['M', 'F'])
            new_person = Person(gender, birth_month=self.current_month, father=father, mother=mother)
            self.population.append(new_person)
            births_count += 1
            # If the mother is still alive, mark her as not pregnant
            if mother in self.population:
                mother.is_pregnant = False
            if self.verbose:
                print(f"  Birth: {new_person} from mother {mother.id} and father {father.id}")
            self.pregnancies.remove(pregnancy)
        return births_count
    
    def process_mating(self):
        for female in self.population:
            if female.gender == 'F' and not female.is_pregnant and female.age(self.current_month) >= 216:
                # Gather eligible males: they must be male, of reproductive age, and not related within the ancestry_limit.
                eligible_males = [
                    male for male in self.population 
                    if male.gender == 'M' and male.age(self.current_month) >= 216 and not related_within(female, male, self.ancestry_limit)
                ]
                if eligible_males:
                    chosen_male = random.choice(eligible_males)
                    female.is_pregnant = True
                    self.pregnancies.append((female, chosen_male, self.current_month))
                    if self.verbose:
                        print(f"  Mating: Female {female.id} mated with Male {chosen_male.id} at month {self.current_month}")
    
    def process_attrition(self):
        if self.population:
            to_remove = random.choice(self.population)
            self.population.remove(to_remove)
            self.pregnancies = [preg for preg in self.pregnancies if preg[0] != to_remove and preg[1] != to_remove]
            if self.verbose:
                print(f"  Attrition: Removed Person {to_remove.id} (Gender: {to_remove.gender}) at month {self.current_month}")

if __name__ == "__main__":
    simulation_months = 600  # 50 years (600 months)
    episodes = 750             # Number of simulation runs per (initial_population, ancestry_limit) combination
    sizes = range(1, 110)    # Testing different initial population sizes
    ancestry_limits = [0, 1e200]  # 0 means ancestral breeding allowed; 1e200 means (practically) no mating with ancestors.
    # We test both because its an interesting question in this population restricted scenario.
    # CSV file configuration (we add an 'episode' field to track progress)
    csv_file_name = 'survival_data.csv'
    csv_header = ['initial_population', 'ancestral_breeding', 'episode', 'time', 'event', 'final_pop']
    
    # Load existing CSV data if present so we can resume from where we left off.
    if os.path.isfile(csv_file_name):
        existing_df = pd.read_csv(csv_file_name)
        print("Existing CSV found. Loaded previously saved simulation data.")
    else:
        existing_df = pd.DataFrame(columns=csv_header)
    
    # Dictionary to store the final population counts (for survival probability calculation).
    results = {(size, limit): [] for size in sizes for limit in ancestry_limits}
    
    # List to hold new survival data generated in this run.
    new_survival_data = []
    
    # Run simulations for each combination. For each (size, limit), only run episodes that haven't been saved yet.
    for limit in ancestry_limits:
        for size in sizes:
            # Map ancestry_limit to binary: 1 if allowed (limit==0), 0 if restricted.
            ancestral_breeding = 1 if limit == 0 else 0
            # Count how many episodes have already been run for this combination.
            already_run = len(existing_df[(existing_df['initial_population'] == size) &
                                          (existing_df['ancestral_breeding'] == ancestral_breeding)])
            # Run only the remaining episodes.
            for episode in range(already_run, episodes):
                print(f"Testing initial population size: {size} for episode: {episode+1} (Ancestry limit: {limit})")
                Person.id_counter = 0  # Reset person ID counter for each simulation episode.
                sim = Simulator(initial_population_size=size, simulation_months=simulation_months, ancestry_limit=limit, verbose=False)
                sim.run()
                results[size, limit].append(sim.population_history[-1])
                
                # Determine time-to-extinction: first month when population becomes 0.
                extinct_month = None
                for month, count in enumerate(sim.population_history):
                    if count == 0:
                        extinct_month = month
                        break
                if extinct_month is None:
                    time_to_extinction = simulation_months
                    event = 0  # censored (no extinction event observed)
                else:
                    time_to_extinction = extinct_month
                    event = 1  # extinction event occurred
                
                # Create a row for this simulation episode.
                row = {
                    'initial_population': size,
                    'ancestral_breeding': ancestral_breeding,
                    'episode': episode + 1,  # using 1-indexed episode number
                    'time': time_to_extinction,
                    'event': event,
                    'final_pop': sim.population_history[-1]
                }
                new_survival_data.append(row)
                
                # Immediately append this row to the CSV file.
                # (The header is written only if the file does not yet exist.)
                file_exists = os.path.isfile(csv_file_name)
                with open(csv_file_name, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
                print("Episode survival data saved to CSV.")
    
    # Combine the previously loaded data with the new data.
    if not existing_df.empty:
        survival_df = pd.concat([existing_df, pd.DataFrame(new_survival_data)], ignore_index=True)
    else:
        survival_df = pd.DataFrame(new_survival_data)
    
    # Compute the survival probability for each combination:
    survival_probabilities = {}
    for limit in [ancestry_limits[1]]:
        for size in sizes:
            ancestral_breeding = 1 if limit == 0 else 0
            group = survival_df[(survival_df['initial_population'] == size) &
                                (survival_df['ancestral_breeding'] == ancestral_breeding)]
            survival_count = sum(1 for pop in group['final_pop'] if pop > 0)
            survival_prob = survival_count / episodes
            survival_probabilities[size, limit] = survival_prob
            print(f"Initial Population {size}: Survival Probability = {survival_prob*100:.1f}%")
    
    # Plot the survival probability vs. initial population size.
    plt.figure(figsize=(10, 6))
    for limit in [ancestry_limits[1]]:
        x_data = sizes
        # True parameters: k=1.0 and x0=5.0; add some noise to simulate measurements.
        y_data = [survival_probabilities[(size, limit)] for size in sizes]
        popt, pcov = curve_fit(logistic, x_data, y_data, p0=[1, 50])
        k_fit, x0_fit = popt
        print(f"Fitted parameters: k = {k_fit:.4f}, x0 = {x0_fit:.4f}")
        plt.plot(x_data, logistic(x_data, k_fit, x0_fit), label=f'Fitted logistic function k = {k_fit:.4f}, x0 = {x0_fit:.4f}')
        
        plt.plot(
            list(sizes),
            [survival_probabilities[(size, limit)] for size in sizes],
            marker='o',
            label="Raw Survival Probabilities from Simulation"
        )
    plt.xlabel("Initial Population Size")
    plt.ylabel("Survival Probability")
    plt.title("Survival Probability vs. Initial Population Size")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # print("\nStatistical significance of initial population (p-value):", p_value_pop)
    print( "Minimum population required", str(np.ceil((-4.59511985013)/-k_fit + x0_fit)))
    # Fit a Cox Proportional Hazards model using the lifelines package.
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col='time', event_col='event', formula="initial_population + ancestral_breeding")
    #cph.print_summary()  # Display coefficients, hazard ratios, and p-values.
    
    # Extract and print the p-values.
    p_value_breed = cph.summary.loc['ancestral_breeding', 'p']
    print("\nStatistical significance of ancestral breeding (p-value):", p_value_breed)
    # p_value_pop = cph.summary.loc['initial_population', 'p']
