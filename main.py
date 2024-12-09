from GA_class import GA

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
    5: (20, 3),
    6: (10, 9),
    7: (4, 6),
    8: (9, 7),
    9: (12, 5),
    10: (16, 4),
}


# population_size = 20
# num_of_generations = 100
# mutuation_rate = 0.1



ga = GA(num_of_customers=len(customers), customers= customers, num_vehicles=5, vehicle_capacity=4, depot_location=(0,0))

best_solution , fitness_score = ga.Evolve()


print(f"Best solution is {best_solution}")
print(f"fitness scoree is {fitness_score}")