from GUI import VehicleRoutingOptimizer
from tkinter import Tk

def main():
    root = Tk()
    app = VehicleRoutingOptimizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()