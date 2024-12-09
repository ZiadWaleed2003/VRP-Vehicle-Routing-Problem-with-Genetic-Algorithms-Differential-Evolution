import numpy as np
import random


class DifferentialEvolution:
    def __init__(self, num_of_customers, customers, num_vehicles, vehicle_capacity,
                 depot_location, min_demand, max_demand, population_size, mutation_rate,
                 crossover_prob, max_iterations):
        
        self.num_of_customers = num_of_customers
        self.customers = customers 
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_prob = crossover_prob
        self.max_iterations = max_iterations
        self.population = self._initialize_population()
        
    def _initialize_population(self):
        """
        Initialize population with customer indices distributed randomly across vehicles,
        ensuring vehicle capacity constraints are respected.
        """
        population = []
        
        for _ in range(self.population_size):
            # Randomly shuffle customer indices
            customer_indices = list(range(self.num_of_customers))
            random.shuffle(customer_indices)

            # Split customers into vehicles while respecting capacity
            routes = [[] for _ in range(self.num_vehicles)]
            vehicle_loads = [0] * self.num_vehicles  # Track current load per vehicle

            for idx in customer_indices:
                customer_demand = self.customers[idx]['demand']
                # Find a vehicle that can accommodate this customer
                valid_vehicles = [
                    v for v in range(self.num_vehicles)
                    if vehicle_loads[v] + customer_demand <= self.vehicle_capacity
                ]

                if valid_vehicles:
                    # Assign the customer to a random valid vehicle
                    vehicle = random.choice(valid_vehicles)
                    routes[vehicle].append(idx)
                    vehicle_loads[vehicle] += customer_demand
                else:
                    # Handle customers that cannot fit into any vehicle (e.g., excessive demand)
                    raise ValueError(f"Customer {idx} with demand {customer_demand} exceeds vehicle capacity constraints.")

            population.append(routes)

        return population

    def fitness_function(self, routes):
            """
            Fitness function: Total distance of all routes.
            """
            total_distance = 0

            for route in routes:
                if len(route) == 0:
                    continue

                current_location = np.array(self.depot_location)
                for customer_id in route:
                    customer_coords = np.array(self.customers[customer_id]['location'])
                    total_distance += np.linalg.norm(customer_coords - current_location)
                    current_location = customer_coords

                total_distance += np.linalg.norm(current_location - np.array(self.depot_location))

            return total_distance
    
    # def _check_capacity(self, route, customer_id):
    #     """
    #     Check if adding the customer to the route exceeds vehicle capacity.
    #     """
    #     current_demand = sum(self.customers[c]['demand'] for c in route)
    #     new_demand = current_demand + self.customers[customer_id]['demand']
    #     return new_demand <= self.vehicle_capacity

    # def mutation(self, target_idx):
    #     """
    #     Mutation function: Generates a mutant vector.
        
    #     Parameters:
    #     - target_idx: Index of the target individual in the population to mutate.

    #     Returns:
    #     - mutant: Mutated individual as a list of routes.
    #     """
    #     # Select three distinct individuals from the population excluding the target
    #     candidates = list(range(self.population_size))
    #     candidates.remove(target_idx)
    #     r1, r2, r3 = random.sample(candidates, 3)

    #     # Extract the routes of the selected individuals
    #     routes_r1 = self.population[r1]
    #     routes_r2 = self.population[r2]
    #     routes_r3 = self.population[r3]

    #     # Create a mutant by combining routes using the mutation formula
    #     mutant = []
    #     for v in range(self.num_vehicles):  # Iterate over vehicles
    #         # Ensure each route has the same length before vector math
    #         max_len = max(len(routes_r1[v]), len(routes_r2[v]), len(routes_r3[v]))
    #         r1_vec = np.pad(routes_r1[v], (0, max_len - len(routes_r1[v])), constant_values=-1)
    #         r2_vec = np.pad(routes_r2[v], (0, max_len - len(routes_r2[v])), constant_values=-1)
    #         r3_vec = np.pad(routes_r3[v], (0, max_len - len(routes_r3[v])), constant_values=-1)

    #         # Apply the mutation formula
    #         mutant_vec = r1_vec + self.mutation_rate * (r2_vec - r3_vec)
            
    #         # Remove padding and invalid indices
    #         mutant_route = [int(customer) for customer in mutant_vec if 0 <= customer < self.num_of_customers]
            
    #         mutant.append(mutant_route)

    #     return mutant

    def mutation(self, target_idx, F=0.8):
        """
        Mutation function: Generate a mutant vector using DE logic.
        Ensures no duplicates, respects capacity constraints, and uses the mutation rate (F).
        """
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3 = random.sample(candidates, 3)

        # Base vector
        base = self.population[r1]
        mutant = [[] for _ in range(self.num_vehicles)]

        # Compute mutation for each vehicle
        for v in range(self.num_vehicles):
            route_r1 = set(base[v])
            route_r2 = set(self.population[r2][v])
            route_r3 = set(self.population[r3][v])

            # Difference vector scaled by F
            difference = route_r2.symmetric_difference(route_r3)  # Compute (r2 - r3) as a symmetric difference
            difference_list = list(difference)  # Convert to list for random sampling
            scaled_difference = set(random.sample(difference_list, int(len(difference_list) * F)))  # Apply F scaling

            # Mutated route: r1 + scaled difference
            mutated_route = route_r1.union(scaled_difference)

            # Filter customers to ensure capacity constraints
            capacity_used = 0
            valid_route = []
            for customer in mutated_route:
                customer_demand = self.customers[customer]['demand']
                if capacity_used + customer_demand <= self.vehicle_capacity:
                    valid_route.append(customer)
                    capacity_used += customer_demand

            mutant[v] = valid_route

        # Ensure all customers are included and no duplicates
        unassigned_customers = set(range(self.num_of_customers)) - set(c for route in mutant for c in route)
        for customer in unassigned_customers:
            for v in range(self.num_vehicles):
                customer_demand = self.customers[customer]['demand']
                current_load = sum(self.customers[c]['demand'] for c in mutant[v])
                if current_load + customer_demand <= self.vehicle_capacity:
                    mutant[v].append(customer)
                    break

        return mutant


    
    def crossover(self, target, mutant):
        """
        Crossover function: Combine target and mutant based on crossover probability.
        """
        trial = []
        for t_route, m_route in zip(target, mutant):
            trial_route = []
            for customer in t_route + m_route:
                if random.random() < self.crossover_prob and customer not in trial_route:
                    trial_route.append(customer)
            trial.append(trial_route)

        return trial
    
    def selection(self, target, trial):
        """
        Selection function: Select the better of target and trial based on fitness.
        """
        target_fitness = self.fitness_function(target)
        trial_fitness = self.fitness_function(trial)
        return trial if trial_fitness < target_fitness else target


    def evolve(self):
        """
        Evolve function: Run the DE algorithm to find the best solution.
        """
        for iteration in range(self.max_iterations):
            new_population = []
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutation(i)
                trial = self.crossover(target, mutant)
                new_individual = self.selection(target, trial)
                new_population.append(new_individual)

            self.population = new_population
            best_solution = min(self.population, key=self.fitness_function)
            print(f"Iteration {iteration + 1}: Best Fitness = {self.fitness_function(best_solution)}")

        return best_solution

    def run(self):
        """
        Run the full DE process, encapsulating evolution and returning the best result.
        """
        print("Starting Differential Evolution...")
        self.evolve()
        # Find the best solution in the population
        best_solution = min(self.population, key=self.fitness_function)
        best_fitness = self.fitness_function(best_solution)

        print(f"Best fitness achieved: {best_fitness}")
        print("Best solution routes:")
        for route in best_solution:
            print(route)
        
        return best_solution, best_fitness
    

if __name__ == "__main__":
    # Parameters
    num_vehicles = 3
    vehicle_capacity = 10
    depot_location = (0, 0)
    population_size = 5
    mutation_rate = 0.8  # F: Mutation factor
    crossover_prob = 0.7  # CR: Crossover probability
    max_iterations = 50
    min_demand = 1
    max_demand = 5
    num_of_customers = 6

    # Test customer data
    customers = {
        0: {'location': (2, 3), 'demand': 2},
        1: {'location': (5, 4), 'demand': 3},
        2: {'location': (8, 1), 'demand': 4},
        3: {'location': (1, 7), 'demand': 5},
        4: {'location': (7, 8), 'demand': 1},
        5: {'location': (3, 3), 'demand': 2}
    }

    # Initialize Differential Evolution Solver
    solver = DifferentialEvolution(
        num_of_customers, customers, num_vehicles, vehicle_capacity, depot_location,
        min_demand, max_demand, population_size, mutation_rate, crossover_prob, max_iterations
    )

    # Run the solver
    best_solution, best_fitness = solver.run()

    # Output best solution
    print("\nFinal Best Solution:")
    print(f"Fitness: {best_fitness}")
    for i, route in enumerate(best_solution):
        print(f"Vehicle {i+1}: {route}")
