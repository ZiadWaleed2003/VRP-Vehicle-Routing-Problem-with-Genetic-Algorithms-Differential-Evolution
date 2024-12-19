import numpy as np
import matplotlib.pyplot as plt

class DE:
    # def __init__(self, num_customers, num_vehicles, capacity, population_size, max_generations=150, cr=0.5, f=0.5, locations = None, depot_coordinates=(0,0), customer_demands = None, max_x = 100, max_y = 100, max_demand = 20, early_stoppimg=50):
    def __init__(self , num_customers, num_vehicles, capacity, f=0.5, depot_coordinates=(0,0), early_stopping = 50):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.f = f
        self.early_stopping = early_stopping
        
        self.cr = 0.1
        self.population_size = 20
        self.max_generations = 100
        self.max_x = 100
        self.max_y = 100
        self.max_demand = capacity
        self.locations = None


        np.random.seed(42)
        self.customer_demands = np.random.rand(num_customers) * self.max_demand
        
        
        self.customer_demands = np.concatenate(([0], self.customer_demands))
        

        self.locations = np.random.rand(num_customers, 2)
        self.locations[:, 0] *= self.max_x  # Multiply first column by max_x
        self.locations[:, 1] *= self.max_y  # Multiply second column by max_y
        
        
        self.locations = np.concatenate(([depot_coordinates[0:]], self.locations), axis=0) # Pass the values as an array

        
        # print(self.locations)
        # print(self.customer_demands)
        # exit()

        self.distances = self.calculate_distances()


    def calculate_distances(self):
        distances = np.zeros((self.num_customers + 1, self.num_customers + 1))
        for i in range(self.num_customers + 1):
            for j in range(i + 1, self.num_customers + 1):
                distances[i, j] = np.linalg.norm(self.locations[i] - self.locations[j])
                distances[j, i] = distances[i, j]
        return distances

    def check_validity(self, trial_solution):
        valid_solution = 0
        start_position = 0
        temp = trial_solution[trial_solution != 0].copy()
        unique = np.unique(temp)
        if len(unique) != len(temp):
            valid_solution += 1
        elif trial_solution.max() > self.num_customers:
            valid_solution += 1
        elif trial_solution.min() < 0:
            valid_solution += 1
        else:
            for _ in range(self.num_vehicles):
                route_start = None
                route_end = None
                count = 0
                for l in range(start_position, self.num_customers + self.num_vehicles + 1):
                    if trial_solution[l] == 0 and route_start is None:
                        route_start = l
                    elif trial_solution[l] == 0 and route_start is not None:
                        route_end = l
                        break
                    else:
                        count += 1
                if route_start is None or route_end is None or count == 0:
                    valid_solution += 1
                route = trial_solution[route_start:route_end+1]
                if np.sum(np.fromiter([self.customer_demands[i] for i in route[route != 0]], float)) > self.capacity:
                    valid_solution += 1
                start_position = route_end
            if start_position != len(trial_solution) - 1:
                valid_solution += 1
        return valid_solution

    def fitness(self, solution):
        total_distance = 0
        routes = np.split(solution, np.where(solution == 0)[0])
        routes = [r for r in routes if len(r) > 0]
        for r in routes:
            r = np.concatenate((r, [0]))
            route_distance = 0
            for i in range(len(r) - 1):
                if r[i] < len(self.distances) and r[i] >= 0 and r[i+1] < len(self.distances) and r[i+1] >= 0:
                    route_distance += self.distances[r[i]][r[i + 1]]
            total_distance += route_distance

        valid_solution = self.check_validity(solution)
        total_distance += ((np.sum(self.distances) * len(self.distances)) / np.count_nonzero(self.distances)) * valid_solution

        return total_distance

    def generate_population(self):
        population = np.zeros((self.population_size, self.num_customers), dtype=int)
        for i in range(self.population_size):
            population[i] = np.arange(1, self.num_customers + 1)

        vehicles = np.zeros((self.population_size, self.num_vehicles - 1), dtype=int)
        population = np.concatenate((population, vehicles), axis=1)

        for solution in population:
            np.random.shuffle(solution)
        zeros = np.zeros((self.population_size, 1), dtype=int)

        population = np.concatenate((population, zeros), axis=1)
        population = np.concatenate((zeros, population), axis=1)

        return population

    def perform_mutation(self, population, current_idx):
        a_idx, b_idx, c_idx = np.random.choice(self.population_size, size=3, replace=False)
        a, b, c = population[a_idx], population[b_idx], population[c_idx]

        current_pop = population[current_idx].copy()
        current_pop = np.delete(current_pop, [0])
        current_pop = np.delete(current_pop, [-1])

        trial_solution = np.zeros(len(current_pop), dtype=int)
        chosen = np.random.randint(self.num_customers + self.num_vehicles - 2)
        values = np.arange(self.num_customers + self.num_vehicles - 2)
        count = 0
        swapped = []
        for k in range(len(current_pop)):
            if k not in swapped:
                if np.random.rand() <= self.cr or k == chosen:
                    if len(values) != 0:
                        value_index = int((a[k] + self.f * (b[k] - c[k])) % (len(values)))
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

    def update_population(self, population, current_idx, trial_solution, trial_fitness, best_fitness, best_solution):
        if trial_fitness < self.fitness(population[current_idx]):
            population[current_idx] = trial_solution
            if trial_fitness < best_fitness:
                best_fitness = trial_fitness
                best_solution = trial_solution
        return population, best_fitness, best_solution

    def differential_evolution(self, print_iter=0, output=False):
        population = self.generate_population()

        best_fitness = np.inf
        for solution in population:
            fitness = self.fitness(solution)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

        fitness_history = []
        for i in range(self.max_generations):
            for j in range(self.population_size):
                trial_solution = self.perform_mutation(population.copy(), j)
                trial_fitness = self.fitness(trial_solution)

                population, best_fitness, best_solution = self.update_population(
                    population, j, trial_solution, trial_fitness, best_fitness, best_solution
                )

            fitness_history.append(best_fitness)
            if output and i % 10**print_iter == 0:
                print(f"Generation {i + 1}/{self.max_generations}: Best fitness = {best_fitness}")

        return best_solution, best_fitness, fitness_history , self.locations