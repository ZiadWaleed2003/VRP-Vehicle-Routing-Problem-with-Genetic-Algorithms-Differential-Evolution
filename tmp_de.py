import numpy as np
import matplotlib.pyplot as plt

# # Define problem parameters
NUM_CUSTOMERS = 5 # Number of customers minus the depot
NUM_VEHICLES = 3
CAPACITY = 9
DE_POPULATION_SIZE = 3
DE_MAX_GENERATIONS = 20
DE_CR = 0.5
DE_F = 0.5

# # Generate random customer demands
np.random.seed(42)
customer_demands = np.random.randint(1, 3, size=NUM_CUSTOMERS)
customer_demands = np.concatenate(([0], customer_demands))
print('Customer Demands: ', customer_demands)

# # Define distance matrix
locations = np.random.rand(NUM_CUSTOMERS + 1, 2)
distances = np.zeros((NUM_CUSTOMERS + 1, NUM_CUSTOMERS + 1))
for i in range(NUM_CUSTOMERS + 1):
    for j in range(i + 1, NUM_CUSTOMERS + 1):
        distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        distances[j, i] = distances[i, j]

def check_validity(trial_solution, customer_demands):
    valid_solution = 0
    start_position = 0
    temp = trial_solution[trial_solution != 0].copy()
    unique = np.unique(temp)
    if len(unique) != len(temp):
        # # Makes sure all customer IDs are unique
        valid_solution += 1
    elif trial_solution.max() > NUM_CUSTOMERS:
        # # Makes sure the largest customer ID is no larger the the total number of customers
        valid_solution += 1
    elif trial_solution.min() < 0:
        # # Makes sure the largest customer ID is greater than 0
        valid_solution += 1
    else: 
        for _ in range(NUM_VEHICLES):
            route_start = None
            route_end = None
            count = 0
            for l in range(start_position, NUM_CUSTOMERS + NUM_VEHICLES + 1):
                if trial_solution[l] == 0 and route_start is None:
                    route_start = l
                elif trial_solution[l] == 0 and route_start is not None:
                    route_end = l
                    break
                else:
                    count += 1
            if route_start is None or route_end is None or count == 0:
                # # Makes sure there is a start and end to each route and each vehicle
                # # visits at least 1 customer
                valid_solution += 1
            route = trial_solution[route_start:route_end+1]
            if np.sum(np.fromiter([customer_demands[i] for i in route[route != 0]], float)) > CAPACITY:
                # # Makes sure the total customer demand for a route isn't greater than the capacity
                # # of a vehicle.
                valid_solution += 1
            start_position = route_end
        if start_position != len(trial_solution) - 1:
            # # Makes sure that the final vehicle return to the depot
            valid_solution += 1
    return valid_solution

# # Define fitness function
def fitness(solution, distances, customer_demands):
    """
    Computes the total distance traveled by all vehicles in the solution, given a list of routes.
    """
    total_distance = 0
    routes = np.split(solution, np.where(solution == 0)[0])
    routes = [r for r in routes if len(r) > 0]
    # total_distance = 0
    for r in routes:
        r = np.concatenate((r, [0]))
        route_distance = 0
        # print(r)
        for i in range(len(r) - 1):
            if r[i] < len(distances) and r[i] >= 0 and r[i+1] < len(distances) and r[i+1] >= 0:
                route_distance += distances[r[i]][r[i + 1]]
        total_distance += route_distance

    # # Check validity of solution
    valid_solution = check_validity(solution, customer_demands)
    total_distance += ((np.sum(distances) * len(distances)) / np.count_nonzero(distances)) * valid_solution

    return total_distance

solution = np.array([0, 5, 1, 0, 2, 0, 4, 3, 0])
total_distance = fitness(solution, distances, customer_demands)
print('The total distance is: ', total_distance)

def generate_population(population_size):
    # # Generate random population, where each individual is a list of customer IDs
    # # This initial population does not include the depot (customer 0)
    # # population = np.random.randint(1, NUM_CUSTOMERS, size=(population_size, NUM_CUSTOMERS))
    population = np.zeros((population_size, NUM_CUSTOMERS), dtype=int)
    for i in range(population_size):
        population[i] = np.arange(1, NUM_CUSTOMERS + 1)

    # # Add the number of returns to the depot so the specified number of vehicles is used
    vehicles = np.zeros((population_size, NUM_VEHICLES - 1), dtype=int)
    population = np.concatenate((population, vehicles), axis=1)

    # # Shuffle the location where each vehicle returns to the depot
    for solution in population:
        solution = np.random.shuffle(solution)
    zeros = np.zeros((population_size, 1), dtype=int)

    # # Add the depot to the beginning and end of each solution
    population = np.concatenate((population, zeros), axis=1)
    population = np.concatenate((zeros, population), axis=1)

    return population

def perform_mutation(population, population_size, current_idx, f, crossover_rate):
    # # Select three random individuals from the population
    a_idx, b_idx, c_idx = np.random.choice(population_size, size=3, replace=False)
    a, b, c = population[a_idx], population[b_idx], population[c_idx]

    # # Select current population member
    current_pop = population[current_idx].copy()
    current_pop = np.delete(current_pop, [0])
    current_pop = np.delete(current_pop, [-1])

    # # Set up trial solution
    trial_solution = np.zeros(len(current_pop), dtype=int)

    # # Perform crossover operation
    chosen = np.random.randint(NUM_CUSTOMERS + NUM_VEHICLES - 2)
    values = np.arange(NUM_CUSTOMERS + NUM_VEHICLES - 2)
    count = 0
    swapped = []
    for k in range(len(current_pop)):
        if k not in swapped:
            if np.random.rand() <= crossover_rate or k == chosen:
                if len(values) != 0:
                    value_index = int((a[k] + f * (b[k] - c[k])) % (len(values)))
                else:
                    break
                swap_index = values[value_index]
                trial_solution[k] = current_pop[swap_index]
                trial_solution[swap_index] = current_pop[k]
                swapped.append(swap_index)
                swapped.append(k)
                values = np.delete(values, value_index)
                values = np.delete(values, np.where(values == k))
            else:
                trial_solution[k] = current_pop[k]

    trial_solution = np.concatenate(([0], trial_solution))
    trial_solution = np.concatenate((trial_solution, [0]))
    return trial_solution

def update_population(population, current_idx, trial_solution, trial_fitness, best_fitness, best_solution, fitness_fn, distances, customer_demands):
    # # Update population with trial solution if it is better
    if trial_fitness < fitness_fn(population[current_idx], distances, customer_demands):
        population[current_idx] = trial_solution
        if trial_fitness < best_fitness:
            best_fitness = trial_fitness
            best_solution = trial_solution
    return population, best_fitness, best_solution

def differential_evolution(fitness_fn, population_size, max_generations, crossover_rate, f, customer_demands, distances, print_iter = 0, output = False):
    # # Initialize population
    population = generate_population(population_size)

    best_fitness = np.inf
    for solution in population:
        fitness = fitness_fn(solution, distances, customer_demands)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution
            
    fitness_history = []
    for i in range(max_generations):
        for j in range(population_size):
            # # Perform mutation and crossover
            trial_solution = perform_mutation(population.copy(), population_size, j, f, crossover_rate)

            trial_fitness = fitness_fn(trial_solution, distances, customer_demands)
                
            # # Update population with trial solution if it is better
            population, best_fitness, best_solution = update_population(population, j, trial_solution, trial_fitness, best_fitness, best_solution, fitness_fn, distances, customer_demands)
        
        fitness_history.append(best_fitness)
        if output:
            if i % 10**print_iter == 0:
                print(f"Generation {i + 1}/{max_generations}: Best fitness = {best_fitness}")

    return best_solution, best_fitness, fitness_history

# # Run Differential Evolution algorithm
best_solution, best_fitness, fitness_history = differential_evolution(
    fitness_fn=fitness,
    population_size=DE_POPULATION_SIZE,
    max_generations=DE_MAX_GENERATIONS,
    crossover_rate=DE_CR,
    f=DE_F,
    customer_demands=customer_demands,
    distances=distances,
    output = True
)
    
# # Print best solution and fitness
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")

# # Visualize fitness history
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness History")
plt.show()