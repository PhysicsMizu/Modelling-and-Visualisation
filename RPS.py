import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import convolve2d

class RPS():
    def __init__(self,size, n_wedges):
        self.size = size
        self.lattice = self.wedges_array()

        # 1 = rock , 2 = paper, 3 = scissors


    def check_nns_ofeachtype(self):
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        #use conv2D to make arrays with the number of neighbours of each type
        rock_nncounts = convolve2d((self.lattice == 1).astype(int) , kernel)
        paper_ascounts = convolve2d((self.lattice == 2).astype(int) , kernel)
        scissors_nncounts = convolve2d((self.lattice == 3).astype(int) , kernel)

        nn_type_counts = np.


        


    def PDupdate(self, nsweeps = 1):



        
    
    def wedges_array(self, wedge_values = np.array([0,1,2], dtype = int)):
        # Create an empty square array
        n_wedges = wedge_values.size()
        array = np.zeros((self.size, self.size))

        # Calculate the center of the array
        center = (self.size - 1) / 2

        # Calculate the angular distance between each wedge
        angle_step = 2 * math.pi / n_wedges

        # Fill the array with wedge values
        for i in range(self.size):
            for j in range(self.size):
                # Calculate the angle of the current position relative to the center
                angle = math.atan2(i - center, j - center) % (2 * math.pi)
                # Calculate the sector index based on the angle
                sector = int(angle / angle_step)
                # Assign the sector value to the array
                array[i, j] = wedge_values[sector]
        return array