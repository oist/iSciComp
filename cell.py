"""Classes for cells"""

import numpy as np
import matplotlib.pyplot as plt

class Cell:
    """Class for a cell"""

    def __init__(self, position = [0,0], radius=0.1, color=[1,0,0,0.5]):
        """Make a new cell"""
        self.position = np.array(position)
        self.radius = radius
        self.color = color
     
    def show(self):
        """Visualize as a circule"""
        c = plt.Circle(self.position,self.radius,color=self.color)
        plt.gca().add_patch(c)
        plt.axis('equal')

if __name__ == "__main__":
    c0 = Cell()
    c0.show()
    plt.show()
