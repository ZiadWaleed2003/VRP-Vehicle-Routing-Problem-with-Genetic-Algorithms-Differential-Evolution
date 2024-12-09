import random 
import numpy as np



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


    def SelectParents(self,fitness_scores , population):

        """
            Select 2 parents using roulette-wheel selection
        """

        total_fitness = sum(fitness_scores)

        probabilities = [score / total_fitness for score in fitness_scores]

        return random.choices(population , weights=probabilities , k=2)
    

    def SingleEdgeCrossOver(self, parent1, parent2):
        """
        Performs Single Edge CrossOver on 2 parents.
        Returns: 
            2 offspring solutions.
        """
        if len(parent1[0]) <= 1 or len(parent2[0]) <= 1:
            # If either parent has an insufficient route, return the parents directly
            return parent1, parent2

        split_points = random.randint(1, len(parent1[0]) - 1)

        offspring1 = [parent1[0][:split_points] + parent2[0][split_points:]]
        offspring2 = [parent2[0][:split_points] + parent1[0][split_points:]]

        return offspring1, offspring2
    

    def swap_mutation(self , solution):

        """
            Perform swap mutation on a solution.
            Returns:
                Mutated solution.
        """

        for route in solution:

            if len(route) > 1 :

                idx1 , idx2 = random.sample(range(len(route)),2)

                route[idx1] , route[idx2] = route[idx2] , route[idx1]

        return solution
    

    def replace_population_with_elitism(self,old_gen , new_gen , depot , retain_fraction=0.1):

        old_gen_scores = [self.FitnessFunction(solution) for solution in old_gen]
        new_gen_scores = [self.FitnessFunction(solution) for solution in new_gen]

        old_gen_sorted = sorted(zip(old_gen_scores , old_gen) , key=lambda x : x[0] , reverse=True)
        new_gen_sorted = sorted(zip(new_gen_scores , new_gen) , key=lambda x : x[0] , reverse=True)


        retain_count = int(len(old_gen) * retain_fraction)

        # Keep the best solution from the old generation (elitism)

        elite_solution = old_gen_sorted[0][1]

        retained_individuals = [elite_solution] + [ind for _, ind in old_gen_sorted[1:retain_count]]

        replacement_individuals = [ind for _, ind in new_gen_sorted[:len(old_gen) - retain_count - 1]]

        combined_gen = retained_individuals + replacement_individuals

        return combined_gen


    

    def Evolve(self):
        """
        Run the genetic algorithm for a specified number of generations.
        Returns:
            Best solution and its fitness.
        """
        
        # initialzing the population 

        population , customer_demands = self.initialize_population()

        # getting the fitness score for the old gen

        for gen in range(self.max_iter):

            fitness_scores = [self.FitnessFunction(solution) for solution in population]

            # Selecting Parents
            new_population = []

            for _ in range(self.population_size // 2): # halfsize cause we want to generate 2 offsprings per iteration

                parent1 , parent2 = self.SelectParents(fitness_scores , population)

                offspring1 , offspring2 = self.SingleEdgeCrossOver(parent1 , parent2)

                # apply conditional mutation 
                
                if random.random() < self.mutation_rate:

                    offspring1 = self.swap_mutation(offspring1)

                if random.random() < self.mutation_rate:

                    offspring2 = self.swap_mutation(offspring2)


                # adding the new gen

                new_population.extend([offspring1 , offspring2])

            
            # replace the old gen with the new gen using elitism

            population = self.replace_population_with_elitism(population , new_population , self.depot_location)

            best_fitness = max([self.FitnessFunction(solution) for solution in population])

            print(f"Generation {gen+1}, Best Fitness: {best_fitness}")


        # return the best solution

        best_solution = max(population , key=self.FitnessFunction)

        return best_solution , self.FitnessFunction(best_solution)    


