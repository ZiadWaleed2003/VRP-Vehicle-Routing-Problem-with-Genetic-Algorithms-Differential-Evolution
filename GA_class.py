import random
import numpy as np

class GA:
    def __init__(self, num_of_customers, customers, num_vehicles, vehicle_capacity, early_stop,depot_location, customer_demands=None , population_size = 10 , max_iter=100 , mutation_rate = 0.5 ):
        # Initialize the GA object with the given parameters
        self.num_of_customers = num_of_customers
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.early_stop = early_stop
        
        # Use the user-provided customer_demands if available
        if customer_demands is not None:
            self.customer_demands = customer_demands
        else:
            raise ValueError("Customer demands must be provided by the user.")

        # Initialize other variables
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initializes the population by randomly generating feasible solutions.

        Each solution is a list of routes, where each route represents the customers served by a vehicle.
        
        Returns:
        - population: List of solutions (each solution is a list of routes).
        """
        population = []

        for _ in range(self.population_size):
            solution = []  # Routes of each vehicle
            available_customers = list(range(1, self.num_of_customers + 1))  # Customer IDs start from 1 to num_of_customers
            random.shuffle(available_customers)  # Shuffle to ensure randomization
            
            # Create empty routes for each vehicle
            vehicle_routes = [[] for _ in range(self.num_vehicles)]
            vehicle_loads = [0] * self.num_vehicles  # Keeps track of total demand on each vehicle

            # **Step 1**: Distribute customers while respecting vehicle capacity
            for customer in available_customers:
                customer_demand = self.customer_demands[customer]
                
                # Find vehicles that have enough capacity to add this customer
                feasible_vehicles = [i for i in range(self.num_vehicles) if vehicle_loads[i] + customer_demand <= self.vehicle_capacity]
                
                if feasible_vehicles:  # If there are feasible vehicles, randomly pick one
                    selected_vehicle = random.choice(feasible_vehicles)
                    vehicle_routes[selected_vehicle].append(customer)
                    vehicle_loads[selected_vehicle] += customer_demand
                else:
                    # **Handle Leftover Customer**: No vehicle has space for this customer, add them to the vehicle with the smallest load
                    min_load_vehicle = min(range(self.num_vehicles), key=lambda i: vehicle_loads[i])
                    vehicle_routes[min_load_vehicle].append(customer)
                    vehicle_loads[min_load_vehicle] += customer_demand

            # **Step 2**: Ensure that no customers are missing or duplicated
            all_customers_assigned = [c for route in vehicle_routes for c in route]
            if len(set(all_customers_assigned)) != self.num_of_customers:
                raise ValueError(f"Some customers were not assigned or were duplicated in routes. Assigned: {set(all_customers_assigned)}")

            solution.extend(vehicle_routes)
            population.append(solution)

        return population


    def calculate_distance(self, route):
        """
        Calculates the total distance of a given route, including the return to the depot.
        
        Parameters:
        - route: List of customers in the order they are visited.
        
        Returns:
        - distance: Total distance of the route.
        """
        distance = 0
        current_location = self.depot_location

        for customer in route:
            customer_location = self.customers[customer]
            distance += np.linalg.norm(np.array(current_location) - np.array(customer_location))
            current_location = customer_location

        # Return to depot
        distance += np.linalg.norm(np.array(current_location) - np.array(self.depot_location))
        return distance

    def fitness_function(self, solution, customer_demands):
        """
        Evaluates the fitness of a solution based on its total distance and capacity constraints.
        
        Parameters:
        - solution: A list of routes (solution).
        - customer_demands: Dictionary of demands for each customer.
        
        Returns:
        - fitness: Fitness value (higher is better).
        """
        total_distance = 0
        for route in solution:
            total_distance += self.calculate_distance(route)

            # Penalize routes that exceed the vehicle's capacity
            if sum(customer_demands[c] for c in route) > self.vehicle_capacity:
                total_distance += 100  # Large penalty for infeasibility

        return 1 / (1 + total_distance)  # Invert distance for maximization

    def select_parents(self, fitness_scores, population):
        """
        Selects two parent solutions from the population using roulette wheel selection.
        
        Parameters:
        - fitness_scores: List of fitness values for each solution.
        - population: List of solutions.
        
        Returns:
        - parent1, parent2: Two selected solutions.
        """
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, weights=probabilities, k=2)

    def order_crossover(self, parent1, parent2):
        """
        Performs Order Crossover (OX) for VRP at the route level.
        Swaps entire routes between two parents, ensuring valid offspring.
        """
        # Ensure both parents have the same number of routes
        max_routes = min(len(parent1), len(parent2)) 
        
        # Pick two random crossover points to swap routes
        if max_routes < 2:
            return parent1, parent2  # No crossover possible if less than 2 routes
        
        cx_point1 = random.randint(0, max_routes - 1)
        cx_point2 = random.randint(cx_point1 + 1, max_routes) if cx_point1 + 1 < max_routes else cx_point1 + 1
        
        # Handle edge case where cx_point1 == cx_point2
        if cx_point1 == cx_point2 or cx_point2 > len(parent1) or cx_point2 > len(parent2):
            return parent1, parent2  # No crossover possible
        
        # Make deep copies of parents (to avoid modifying originals)
        offspring1 = parent1[:]
        offspring2 = parent2[:]

        # Swap the routes between crossover points
        offspring1[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        offspring2[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        
        # Ensure customer uniqueness by fixing duplicates and missing customers
        offspring1 = self.repair_solution(offspring1)
        offspring2 = self.repair_solution(offspring2)
        
        return offspring1, offspring2
    
    def repair_solution(self, solution):
        """
        Ensures that each customer is included exactly once in the solution.
        Removes duplicates and adds missing customers to under-loaded routes.
        """
        # Flatten the solution to track all customers in all routes
        all_customers_in_solution = [customer for route in solution for customer in route]
        all_customers_in_solution_set = set(all_customers_in_solution)

        # Get the list of missing customers
        all_customers_set = set(range(1, self.num_of_customers + 1))  # Assuming customer IDs start from 1 to num_of_customers
        missing_customers = list(all_customers_set - all_customers_in_solution_set)  # Customers not present in the solution

        # Remove duplicates while maintaining capacity constraints
        for route in solution:
            seen = set()
            route[:] = [customer for customer in route if not (customer in seen or seen.add(customer))]

        # Add missing customers back into the least-loaded routes
        for customer in missing_customers:
            # Find the route with the least number of customers
            least_loaded_route = min(solution, key=lambda x: len(x))
            least_loaded_route.append(customer)

        return solution

    def swap_mutation(self, solution):
        """
        Applies swap mutation by randomly swapping two customers in each route.
        
        Parameters:
        - solution: A list of routes.
        
        Returns:
        - solution: Mutated solution.
        """
        for route in solution:
            if len(route) > 1:
                idx1, idx2 = random.sample(range(len(route)), 2)
                route[idx1], route[idx2] = route[idx2], route[idx1]
        return solution

    def enforce_capacity(self, solution, customer_demands):
        """
        Enforces capacity constraints by removing customers from routes that exceed capacity.
        
        Parameters:
        - solution: A list of routes.
        - customer_demands: Dictionary of demands for each customer.
        
        Returns:
        - solution: Adjusted solution with feasible routes.
        """
        for route in solution:
            while sum(customer_demands[c] for c in route) > self.vehicle_capacity:
                # Remove a random customer exceeding capacity
                route.pop(random.randint(0, len(route) - 1))
        return solution

    def evolve(self):
        """
        Runs the Genetic Algorithm to optimize the solution for the VRP.
        
        Returns:
        - best_solution: The best solution found.
        - best_fitness: Fitness value of the best solution.
        """
        # Initialize population and customer demands
        population = self.initialize_population()

        best_fitness = float('-inf')  # Track the best fitness score
        best_gen = None
        no_improvement_counter = 0  # Counter to track generations without improvement
        fitness_collec = []
        pop_collec = []

        for gen in range(self.max_iter):
            fitness_scores = [self.fitness_function(sol, self.customer_demands) for sol in population]
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(fitness_scores, population)
                offspring1, offspring2 = self.order_crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    offspring1 = self.swap_mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self.swap_mutation(offspring2)

                offspring1 = self.enforce_capacity(offspring1, self.customer_demands)
                offspring2 = self.enforce_capacity(offspring2, self.customer_demands)

                new_population.extend([offspring1, offspring2])

            # Retain diversity by removing duplicates
            unique_population = list({str(sol): sol for sol in new_population}.values())
            population = unique_population[:self.population_size]

            current_fitness = max([self.fitness_function(sol, self.customer_demands) for sol in population])
            print(f"Generation {gen + 1}, Best Fitness: {current_fitness}")

            pop_collec.append(population)
            fitness_collec.append(current_fitness)

            if current_fitness > best_fitness:

                best_fitness = current_fitness # change the best fitness to the current one
                best_gen     = population
                no_improvement_counter = 0  # reset the counter

            else:
                no_improvement_counter += 1 # increment the counter

            
            if no_improvement_counter >= self.early_stop:

                print(f"Early stopping at generation {gen + 1} due to no improvement.")
        
                break



        idx = fitness_collec.index(max(fitness_collec))
        best_gen = pop_collec[idx]
        best_fitness = fitness_collec[idx]

        best_solution = max(best_gen, key=lambda sol: self.fitness_function(sol, self.customer_demands))

        return best_solution , best_fitness , idx




