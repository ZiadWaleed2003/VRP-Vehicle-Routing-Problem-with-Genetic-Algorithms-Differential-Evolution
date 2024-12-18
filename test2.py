# Generate random customer locations and demands
import random
import numpy as np
from GA_class import GA
import matplotlib.pyplot as plt


num_customers = 50
customers = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(1, num_customers + 1)}
customer_demands = {i: random.randint(1, 10) for i in range(1, num_customers + 1)}
num_vehicles = 5
vehicle_capacity = 50
depot_location = (50, 50)

# Run GA with larger dataset
ga_large = GA(
    num_of_customers=num_customers,
    customers=customers,
    num_vehicles=num_vehicles,
    vehicle_capacity=vehicle_capacity,
    depot_location=depot_location,
    customer_demands=customer_demands,
    population_size=30,
    max_iter=100,
    mutation_rate=0.3,
    early_stop=10
)

best_solution_large, best_fitness_large , idx = ga_large.evolve()

# Output results
print(f"Best Solution (Large) : {best_solution_large}")
print(f"Best Fitness (Large) at {idx+1}:  {best_fitness_large}")


plt.figure(figsize=(10,10))
for route in best_solution_large:
    x, y = [customers[c][0] for c in route], [customers[c][1] for c in route]
    plt.plot([0] + x + [0], [0] + y + [0], marker='o', label=f"Route {route}")

plt.legend(loc='upper left')
plt.title("Vehicle Routes")
plt.show()
