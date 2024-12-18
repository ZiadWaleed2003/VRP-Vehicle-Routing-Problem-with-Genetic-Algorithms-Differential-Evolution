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



# # Example usage
# if __name__ == "__main__":
    
#     num_customers = int(input("Enter the number of customers: "))
#     num_vehicles = int(input("Enter the number of vehicles: "))
#     capacity = int(input("Enter the vehicle capacity: "))
#     DE_F = float(input("Enter the scaling factor (f): "))
#     early_stopping = int(input("Enter the early stopping value: "))
#     # max_demand = capacity    
#     # max_x = 30
#     # max_y = 30
#     # population_size = int(input("Enter the population size: "))
#     # max_generations = int(input("Enter the maximum number of generations: "))
#     # DE_CR = float(input("Enter the crossover rate (cr): "))

#     while True:
#       try:
#           depot_x, depot_y = map(float, input("Enter depot coordinates (x, y): ").split(","))
#           if depot_x >= 0 and depot_y >= 0:
#               depot_coordinates = (depot_x, depot_y)
#               break
#           else:
#               print("Depot coordinates must be non-negative. Please try again.")
#       except ValueError:
#           print("Invalid input format. Please enter coordinates as x,y.")


#     locations = None
#     # enter_locations = input("Do you want to enter customer locations? (yes/no): ")
#     enter_locations = "no"

#     if enter_locations[0].lower() == "y":
#             locations = list()

#             print("Enter customer locations (x, y coordinates):")
#             print(f"# X range: [0 .. {max_x}]")
#             print(f"# Y range: [0 .. {max_y}]")
#             for i in range(num_customers):
#                 while True:
#                     try:
#                         x, y = map(float, input(f"Customer {i + 1}: ").split(","))
#                         if x >= 0 and x <= max_x and y >= 0 and y <= max_y:
#                             locations.append([x, y])  # Append to locations list
#                             break
#                         else:
#                             print(f"Demand value must be in valid range x[0 .. {max_x}]  y[0 .. {max_y}]. Please try again.")
#                     except ValueError:
#                         print("Invalid input format. Please enter coordinates as x,y.")

#             locations = np.array(locations)  # Convert to NumPy array



#     customer_demands = None
#     # enter_locations = input("Do you want to enter customer demands? (yes/no): ")
#     enter_locations = "no"

#     if enter_locations[0].lower() == "y":
#             customer_demands = list()

#             print("Enter customer demans as an ( numerical value ):")
#             print(f"# demands range: (0 .. {max_demand}]")

#             for i in range(num_customers):
#                 while True:
#                     try:
#                         x = float(input(f"Customer {i + 1}: "))
#                         if x > 0 and x <= max_demand:
#                             customer_demands.append(x)  # Append to locations list
#                             break
#                         else:
#                             print(f"Demand value must be in valid range (0 .. {max_demand}]. Please try again.")
#                     except ValueError:
#                         print("Invalid input format. Please try again a numerical value")

#             customer_demands = np.array(customer_demands)  # Convert to NumPy array
    
#     if customer_demands is not None and np.any(customer_demands) > capacity:
#         print("Error: Customer demands cannot exceed the vehicle capacity.")
#         exit()

    
#     de = DE(
#         num_customers       = num_customers,
#         num_vehicles        = num_vehicles,
#         capacity            = capacity,
#         f                   = DE_F,
#         depot_coordinates   = depot_coordinates,
#         early_stopping      = early_stopping
#         # population_size     = population_size,
#         # max_generations     = max_generations,
#         # cr                  = DE_CR,
#         # locations           = locations,
#         # customer_demands    = customer_demands,
#         # max_x               = max_x,
#         # max_y               = max_y,
#         # max_demand          = max_demand
#     )

#     best_solution, best_fitness, fitness_history = de.differential_evolution(output=True)

#     if locations is None:
#       locations = de.locations

#     print(f"Best solution: {best_solution}")
#     print(f"Best fitness: {best_fitness}")

#     plt.plot(fitness_history)
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness")
#     plt.title("Fitness History")
#     plt.show()
    

# ######################################
# ## Graphical representation 
# ######################################

# import matplotlib as mpl
# from matplotlib.patches import Patch

# def plot_cvrp_solution(locations, solution):
    
#     # # Define a color map to use for the routes
#     cmap = mpl.colormaps['hsv']

#     # # Create a dictionary to store the colors for each route
#     color_dict = {}
    
#     # # Create a plot and set the plot size
#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     # # Plot the customer locations
#     ax.scatter([loc[0] for loc in locations], [loc[1] for loc in locations], s=100, color='black')

#     # # Get the solution vehicle routes
#     routes = np.split(solution, np.where(solution == 0)[0])
#     routes.pop(0)
#     routes.pop(-1)
    
#     # # Plot the solution routes
#     for i in range(len(routes)):
#         route = routes[i]
#         route = np.concatenate((route, [0]))

#         print(route)

#         color = cmap(i / len(routes))
        
#         # # Create a line plot for the route
#         ax.plot([locations[x][0] for x in route], [locations[x][1] for x in route], color=color, linewidth=3, label=f'Vehicle {i}')

#         color_dict[f"Route {i}"] = color
        
    
#     # # Set the axis limits and labels
#     ax.set_xlim([0, max([loc[0] for loc in locations]) * 1.1])
#     ax.set_ylim([0, max([loc[1] for loc in locations]) * 1.1])
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
    
#     # # Set the title
#     ax.set_title(f'CVRP Solution ({num_customers} Customers, {num_vehicles} Vehicles, Capacity {capacity})')

#     # # Create a legend for the solution routes
#     legend_handles = [Patch(facecolor=color_dict[label], label=label) for label in color_dict.keys()]

#     # # Define the coordinates for the legend box
#     legend_x = 1
#     legend_y = 0.5
    
#     # # Place the legend box outside of the graph area
#     plt.legend(handles=legend_handles, bbox_to_anchor=(legend_x, legend_y), loc='center left', title='Routes')
    
#     # # Show the plot
#     plt.show()

# # # Plot graph with solution
# plot_cvrp_solution(locations, best_solution)