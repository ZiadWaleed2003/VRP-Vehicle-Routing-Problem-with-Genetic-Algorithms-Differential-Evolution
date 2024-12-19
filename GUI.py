from tkinter import Tk, Label, Entry, Button, Radiobutton, IntVar, Frame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from GA_class import GA
from DE_class import DE
import random



class VehicleRoutingOptimizer:
    def __init__(self, root):
        self.app = root
        self.app.geometry("1400x920")
        self.app.title("Vehicle Routing Problem Optimization")
        self.app.resizable(False, False)
        self.app.configure(bg="#0E0E0E")
        self.algo_selected = None
        self.fitness_text = None
        # Set up layout frames
        self.left_frame = Frame(self.app, width=1100, height=900, bg="#1E1E1E")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.right_frame = Frame(self.app, width=300, height=900, bg="#1E1E1E")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Ensure the frames scale with the window
        self.app.grid_columnconfigure(0, weight=3)
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_rowconfigure(0, weight=1)

        # Add the graph section
        self.fig, self.ax = plt.subplots(figsize=(10, 8), facecolor="white")
        self.ax.set_facecolor("white")
        self.ax.set_title("Vehicle Routes", color="black", fontsize=16)
        self.ax.tick_params(colors="black")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Initialize placeholder customers and depot
        self.depot = [0, 0]
        self.customers = {
            1: [2, 3],
            2: [5, 6],
            3: [1, 7],
            4: [4, 2],
            5: [3, 5]
        }

        # Add input widgets to the right frame
        self.create_input_widgets()

    def GA_selected(self):
        self.algo_selected = 'GA'
        print("GA is selected")

    def DE_selected(self):
        self.algo_selected = 'DE'
        print("DE is selected")

    def create_input_widgets(self):
        # Algorithm Selection
        algorithm_label = Label(self.right_frame, text="Algorithm", bg="#1E1E1E", fg="white", font=("Helvetica", 14))
        algorithm_label.pack(pady=(20, 10))
        self.selected_algorithm = IntVar()
        self.fitness_label = Label(self.right_frame, text="Fitness Function Value: N/A", bg="#1E1E1E", fg="white", font=("Helvetica", 12))
        self.fitness_label.pack(pady=(25, 15))

        ga_radio = Radiobutton(self.right_frame, text="Genetic Algorithm", variable=self.selected_algorithm, value=1, 
                                bg="#1E1E1E", fg="white", selectcolor="#1E1E1E", font=("Helvetica", 12) , command=self.GA_selected)
        ga_radio.pack(pady=10)
        de_radio = Radiobutton(self.right_frame, text="Differential Evolution", variable=self.selected_algorithm, value=2, command=self.DE_selected ,  
                                bg="#1E1E1E", fg="white", selectcolor="#1E1E1E", font=("Helvetica", 12))
        de_radio.pack(pady=10)

        # Input Fields
        fields = [
            ("Number of customers", "5"),
            ("Number of vehicles", "2"),
            ("Vehicles Capacity", "50"),
            ("Depot Location", "0, 0"),
            ("Mutation Rate", "0.1"),
            ("Early Stopping", "100"),
        ]
        self.entries = {}
        for label_text, default_value in fields:
            label = Label(self.right_frame, text=label_text, bg="#1E1E1E", fg="white", font=("Helvetica", 12))
            label.pack(pady=(15, 5))
            entry = Entry(self.right_frame, bg="#2E2E2E", fg="white", insertbackground="white", font=("Helvetica", 12))
            entry.insert(0, default_value)
            entry.pack(pady=10, ipady=5, ipadx=5)
            self.entries[label_text] = entry

        # Run Optimization Button
        run_button = Button(self.right_frame, text="Run Optimization", command=self.run_optimization, 
                            bg="#0078D7", fg="white", activebackground="#005A9E", 
                            activeforeground="white", font=("Helvetica", 14), width=20, height=2)
        run_button.pack(pady=(30, 20))

   

    def run_optimization(self):
        # Validate inputs
        try:
            num_customers = int(self.entries["Number of customers"].get())
            num_vehicles = int(self.entries["Number of vehicles"].get())
            vehicle_capacity = int(self.entries["Vehicles Capacity"].get())
            depot_location = list(map(float, self.entries["Depot Location"].get().split(',')))
            mutation_rate = float(self.entries["Mutation Rate"].get())
            early_stopping = int(self.entries["Early Stopping"].get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values")
            return

        random.seed(42)
        np.random.seed(42)

        if self.algo_selected == 'GA':

            
           
            customers = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(1, num_customers + 1)}
            customer_demands = {i: random.randint(1, 10) for i in range(1, num_customers + 1)}
        
            ga = GA(
                num_of_customers = num_customers,
                customers = customers,
                num_vehicles = num_vehicles,
                vehicle_capacity = vehicle_capacity,
                depot_location = depot_location,
                customer_demands = customer_demands,
                mutation_rate = mutation_rate,
                early_stop = early_stopping
            )
                
            best_sol , best_fitness , idx = ga.evolve()
            self.fitness_text = best_fitness

            # Plot the routes
            self.plot_routes(best_sol , customers=customers , depot = depot_location )
        
        elif self.algo_selected == 'DE':

            customers = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(1, num_customers + 1)}
            customer_demands = {i: random.randint(1, 10) for i in range(1, num_customers + 1)}

            # DE main function here
            de = DE(
                    num_customers       = num_customers,
                    num_vehicles        = num_vehicles,
                    capacity            = vehicle_capacity,
                    f                   = mutation_rate,
                    depot_coordinates   = depot_location,
                    early_stopping      = early_stopping
                )
            
            best_solution, best_fitness, fitness_history , locations = de.differential_evolution(output=True)
            self.fitness_text = best_fitness

            best_sol = []
            result = []
            for value in best_solution:
                if value == 0:
                    # Add a new row when a zero is encountered
                    result.append([])
                else:
                    # Add the value to the last row
                    if result:  # Ensure there's at least one row to append to
                        result[-1].append(value)
                    else:
                        result.append([value])  # Handle the case when the array starts with non-zero

            # Remove the first and last rows
            if len(result) > 1:  # Ensure there are at least two rows
                result = result[:-1]


            self.plot_routes(result , customers=locations , depot = depot_location )


            print(f"Best solution: {best_solution}")
            print(f"Best fitness: {best_fitness}")

        else:

            print('Place holder')


    def plot_routes(self, routes , customers , depot):
        # Clear previous plot
        self.ax.clear()
        self.ax.set_title("Vehicle Routes", color="black", fontsize=16)
        self.ax.set_facecolor("white")
        self.ax.tick_params(colors="black")

        plt.figure(figsize=(10,10))
        for route in routes:
            x, y = [customers[c][0] for c in route], [customers[c][1] for c in route]
            # Use depot coordinates instead of [0,0]
            self.ax.plot([depot[0]] + x + [depot[0]], [depot[1]] + y + [depot[1]], 
                        marker='o', label=f"Route {route}")

        # Plot depot
        self.ax.scatter(depot[0], depot[1], color='red', s=200, marker='s', label='Depot')

        # Customize the plot
        self.ax.legend(facecolor="white", edgecolor="black", labelcolor="black", fontsize=8)
        self.ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

        fitness = f"Fitness Function Value: {self.fitness_text}"
        self.fitness_label.config(text=fitness)

        # Redraw the canvas
        self.canvas.draw()


