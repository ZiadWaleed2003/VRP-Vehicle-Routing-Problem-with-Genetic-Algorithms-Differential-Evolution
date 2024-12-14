from tkinter import Tk, Label, Entry, Button, Radiobutton, IntVar, Frame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class VehicleRoutingOptimizer:
    def __init__(self, root):
        self.app = root
        self.app.geometry("1400x920")
        self.app.title("Vehicle Routing Problem Optimization")
        self.app.resizable(False, False)
        self.app.configure(bg="#0E0E0E")

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

    def create_input_widgets(self):
        # Algorithm Selection
        algorithm_label = Label(self.right_frame, text="Algorithm", bg="#1E1E1E", fg="white", font=("Helvetica", 14))
        algorithm_label.pack(pady=(20, 10))
        self.selected_algorithm = IntVar()
        ga_radio = Radiobutton(self.right_frame, text="Genetic Algorithm", variable=self.selected_algorithm, value=1, 
                                bg="#1E1E1E", fg="white", selectcolor="#1E1E1E", font=("Helvetica", 12))
        ga_radio.pack(pady=10)
        de_radio = Radiobutton(self.right_frame, text="Differential Evolution", variable=self.selected_algorithm, value=2, 
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
            vehicle_capacity = float(self.entries["Vehicles Capacity"].get())
            depot_location = list(map(float, self.entries["Depot Location"].get().split(',')))
            mutation_rate = float(self.entries["Mutation Rate"].get())
            early_stopping = int(self.entries["Early Stopping"].get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values")
            return

        # Placeholder optimization result
        best_solution = [
            [1, 3, 5],  # Route 1
            [2, 4]      # Route 2
        ]

        # Plot the routes
        self.plot_routes(best_solution)

    def plot_routes(self, routes):
        # Clear previous plot
        self.ax.clear()
        self.ax.set_title("Vehicle Routes", color="black", fontsize=16)
        self.ax.set_facecolor("white")
        self.ax.tick_params(colors="black")

        # Color palette for routes
        colors = ['blue', 'orange', 'green', 'red', 'purple']

        # Plot each route
        for i, route in enumerate(routes):
            # Get x and y coordinates for the route, including depot at start and end
            x = [self.depot[0]] + [self.customers[c][0] for c in route] + [self.depot[0]]
            y = [self.depot[1]] + [self.customers[c][1] for c in route] + [self.depot[1]]
            
            # Plot the route
            self.ax.plot(x, y, marker='o', color=colors[i % len(colors)], 
                         linewidth=2, label=f"Route {i+1}")
            
            # Plot customer points
            self.ax.scatter([self.customers[c][0] for c in route], 
                            [self.customers[c][1] for c in route], 
                            color=colors[i % len(colors)], s=100)

        # Plot depot
        self.ax.scatter(self.depot[0], self.depot[1], color='red', s=200, marker='s', label='Depot')

        # Customize the plot
        self.ax.legend(facecolor="white", edgecolor="black", labelcolor="black", fontsize=10)
        self.ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        
        # Redraw the canvas
        self.canvas.draw()

def main():
    root = Tk()
    app = VehicleRoutingOptimizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

