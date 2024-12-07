import random 
import numpy as np

# idk if we are gonna use a DB or terminal this is a first draft anyways
# num_customers = 10
# max_vehicles = 5
# min_demand = 1
# max_demand = 8

customers = {
    1: (2, 3),
    2: (5, 8),
    3: (7, 2),
    4: (4, 6),
    5: (8, 3)
}


# population_size = 20
# num_of_generations = 100
# mutuation_rate = 0.1


class GA:
    def __init__(self, num_of_customers, customers, num_vehicles, vehicle_capacity, depot_location, min_demand=1, max_demand=5,
                 population_size=50, mutation_rate=0.1, max_iter=100):
        self.num_of_customers = num_of_customers
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.max_iter = max_iter
        self.population = []

    def initialize_population(self):
        population = []
        customer_demands = {customer: random.randint(self.min_demand, self.max_demand) for customer in range(1, self.num_of_customers + 1)}

        for _ in range(self.population_size):
            solution = []  #routes of each vehicle
            available_customers = list(range(1, self.num_of_customers + 1)) 

            for _ in range(self.num_vehicles):
                num_customers_in_vehicle = random.randint(1, len(available_customers))
                # Randomly select cxs for this vehicle's route
                route = random.sample(available_customers, num_customers_in_vehicle)
                # Remove selected cxs from the available pool
                available_customers = [cx for cx in available_customers if cx not in route]
                solution.append(route) 
                if not available_customers:
                    break
            # any remaining cxs will be assigned in last vehicle
            if available_customers:
                solution[-1].extend(available_customers)
            population.append(solution)
        return population, customer_demands

    def CalculateDistance(self, route):
        distance = 0
        current_location = self.depot_location

        for customer in route:
            customer_location = self.customers[customer]
            distance += np.linalg.norm(np.array(current_location) - np.array(customer_location))
            #set current location to current customer 
            current_location = customer_location
        distance += np.linalg.norm(np.array(current_location) - np.array(self.depot_location))

        return distance
    
    def FitnessFunction(self, solution):
        total_distance = 0
        for route in solution:
            total_distance += self.CalculateDistance(route)
        return 1 / (1 + total_distance) # total distance inc. fitness dec.





ga_instance = GA(num_of_customers=5, customers= customers, num_vehicles=3, vehicle_capacity=20, depot_location=(0,0))

population, customer_demands = ga_instance.initialize_population()


print(f"Population: {population}")
print(f"Customer Demands: {customer_demands}")

fitness = ga_instance.FitnessFunction(population[0])

print(f"Fitness of first solution is : ", fitness)

