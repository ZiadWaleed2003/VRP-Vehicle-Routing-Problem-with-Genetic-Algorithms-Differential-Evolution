from GA_class import GA
import random
import numpy as np

# Test Data
customers = {
    1: (2, 3),
    2: (5, 8),
    3: (7, 2),
    4: (4, 6),
    5: (20, 3),
    6: (10, 9),
    7: (4, 6),
    8: (9, 7),
    9: (12, 5),
    10: (16, 4),
}

# Hardcoded customer demands (for now)
customer_demands = {
    1: 5,
    2: 3,
    3: 6,
    4: 2,
    5: 8,
    6: 4,
    7: 7,
    8: 3,
    9: 5,
    10: 6,
}

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize the GA
ga = GA(
    num_of_customers=len(customers),
    customers=customers,
    num_vehicles=5,
    vehicle_capacity=8,  # Adjusted to match potential demands
    depot_location=(0, 0),
    customer_demands= customer_demands
)

# Run the algorithm
best_solution, fitness_score = ga.evolve()

# Print the best solution and its fitness
print(f"Best solution is: {best_solution}")
print(f"Fitness score is: {fitness_score}")