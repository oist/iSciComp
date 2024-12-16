"""Classes for cells"""

import numpy as np
import matplotlib.pyplot as plt
import cell

class gCell(cell.Cell):
    """Class of growing cell based on Cell class"""
    
    def grow(self, scale=2):
        """Grow the area of the cell"""
        self.radius *= np.sqrt(scale)
        
    def duplicate(self):
        """Make a copy with a random shift"""
        c = gCell(self.position+np.random.randn(2)*self.radius, self.radius, self.color)
        return c

if __name__ == "__main__":
    c0 = gCell()
    c0.show()
    c1 = c0.duplicate()
    c1.grow()
    c1.show()
    plt.show()
