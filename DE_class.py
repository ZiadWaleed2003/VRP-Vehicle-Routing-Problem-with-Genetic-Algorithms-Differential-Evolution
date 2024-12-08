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

        population = []
        
        for _ in range(self.population_size):
            # Shuffle customers randomly
            customer_indices = list(range(self.num_of_customers))
            random.shuffle(customer_indices)
            
            # Split customers across the available vehicles randomly
            routes = [[] for _ in range(self.num_vehicles)]
            remaining_demand = {i: random.randint(self.min_demand, self.max_demand) for i in range(self.num_of_customers)}

            for customer in customer_indices:
                # Randomly select a vehicle to add this customer to, ensuring vehicle capacity isn't exceeded
                assigned = False
                while not assigned:
                    vehicle = random.randint(0, self.num_vehicles - 1)  # Randomly select a vehicle
                    if sum(remaining_demand[c] for c in routes[vehicle]) + remaining_demand[customer] <= self.vehicle_capacity:
                        routes[vehicle].append(customer)
                        assigned = True
            population.append(routes)

        return population

if __name__ == "__main__":
    # Example input values
    num_vehicles = 3
    vehicle_capacity = 15
    depot_location = (0, 0)
    population_size = 20
    mutation_rate = 0.8
    crossover_prob = 0.7
    max_iterations = 50
    min_demand=1
    max_demand=5
    num_of_customers=6
    # Predefined customer data
    customers = {
        0: {'location': (2, 3), 'demand': 2},
        1: {'location': (5, 4), 'demand': 3},
        2: {'location': (8, 1), 'demand': 1},
        3: {'location': (1, 7), 'demand': 4},
        4: {'location': (7, 8), 'demand': 2},
        5: {'location': (3, 3), 'demand': 3}
    }

    # Initialize DE for solving VRP
    solver = DifferentialEvolution(
        num_of_customers,customers,num_vehicles, vehicle_capacity, depot_location,min_demand,max_demand, population_size,
        mutation_rate, crossover_prob, max_iterations)
    check=solver._initialize_population()
    print(check)

    # Run the optimiz