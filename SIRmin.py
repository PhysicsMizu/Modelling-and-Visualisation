import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from tqdm import tqdm 
import time
from scipy.signal import convolve2d as convolve2d

class SIRS():
    
    """Class to call on the SIRS model"""
    def __init__(self , size, p1, p2, p3, permanent_immunity = 0):
        self.size = size
        self.sweep = self.size**2
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.permanent_immunity= permanent_immunity

        if self.permanent_immunity ==0:
            self.lattice = np.random.choice([0,1,2], size= (self.size , self.size))
            #Susceptible = 0 , Infected = 2 , Recovered = 1     
        elif self.permanent_immunity > 0:
            p_equalforothers = (1-self.permanent_immunity)/3
            self.lattice = np.random.choice([0,1,2, 3], size= (self.size , self.size), p= [p_equalforothers, p_equalforothers,p_equalforothers,self.permanent_immunity ])
            #Susceptible = 0 , Infected = 2 , Recovered = 1 , permanently immune = 3
        
    def clear_history(self):
        """Clears history by reinitializing the lattice but keeps the original lattice size and probabilities"""
        self.__init__(self.size, self.p1, self.p2, self.p3, self.permanent_immunity)

        
    def clear_history_and_change_p(self, newp1, newp2, newp3, new_permimmune = 0):
        """Clears history by reinitializing the lattice. It also changes the probabilities p1, p2, p3 (these should be floats)"""
        self.__init__(self.size, newp1, newp2, newp3 , new_permimmune)
            
            
    def n_infected_neighbours(self,point):
            """given point tuple of indices (i,j) , 
            returns the number of infected neighbours to that point in self.lattice"""
            i, j = point
            above = (i - 1)%self.size
            below = (i + 1)%self.size
            right = (j + 1)%self.size
            left = (j - 1)%self.size
            S = 0
            neighbours = [self.lattice[above , j] ,self.lattice[below , j], \
                         self.lattice[i , right] , self.lattice[i , left] ]
            return (2 in neighbours) 

    def update (self, sweeps = 1):
        """Attempt to update 'runs' randomly chosen points in the lattice
        (RANDOM SEQUENTIAL)"""
        if self.count_infected() == 0:
            return 
        for _ in range(sweeps): 
            random_cells = np.random.randint(0, self.size   , size = (self.sweep, 2))
            for i in range(self.sweep):
                # if self.count_infected() == 0:
                #     return 
                point = random_cells[i,0] , random_cells[i,1]  #random.randint(-1,self.size-2) , random.randint(-1,self.size-2)
                if self.lattice[point] == 0:
                    if self.n_infected_neighbours(point) == True:
                        self.lattice[point] =  random.choices([2, 0], weights= [self.p1 , 1 -self.p1 ])[0]
                elif self.lattice[point] == 2:
                    self.lattice[point] =  random.choices([1, 2], weights = [self.p2 , 1 -self.p2 ])[0]
                elif self.lattice[point] == 1:
                    self.lattice[point] =  random.choices([0, 1], weights = [self.p3 , 1 -self.p3 ])[0]

    def count_infected(self):
        return self.lattice[self.lattice == 2].size
    
    
    # def bootstrap_sorting(self, data_array, n =  1000):
    #     """ Bootstrap sampling algorithm for an array of length i: returns n samples of size i in an array 
    #     of size (i,n)"""
    
    #     rows = data_array.shape[0]

    #     random_selection = np.random.randint(0, n , size = (rows, n))
    #     #make the sets
    #     sets = np.zeros((rows, n))
    #     for i in range(rows):
    #         for k in range(n):
    #                 sets[i,k] = data_array[random_selection[i,k]]
    #     #Take the std of the function over the sets
    #     return sets
    
    def gather_and_measure_sample(self, t_equil , t_decorrel, sample_size=1):
        """ Given floats t_equil , t_decorrel and a sample size, returns <I> and varI for the sample"""
        I = np.zeros(sample_size)
        self.clear_history()
        self.update(sweeps=t_equil)
        for i in range(sample_size):
            I[i] = self.count_infected()
            self.update(sweeps= t_decorrel)
        I = I/self.sweep
        mean_I , var_I = np.mean(I) , np.var(I)
        return mean_I , var_I

    
    def p1p3_phase_diagram(self, resolution=0.05, sample_size = 100, t_equil=100 , t_decorrel=1, filename = 'SIRS_PHASE_DIAGRAM_ASARRAYS'):

        """given (min, max) tuples for the range of p1 , p2, p3 you want to explore , this function creates an array of p1, p2, p3 combinations and measures I and """
        p1_array = np.linspace(0, 1, int((1/resolution)+1))
        p3_array = np.linspace(0, 1, int((1/resolution)+1))


        DATA_AS_ARRAY = np.empty((p1_array.size , p3_array.size, 2))
        len1 = p1_array.size
        len3 = p3_array.size

        for i in tqdm(range(len1)):
            for j in tqdm(range(len3)):
                p1 , p2, p3 = p1_array[i], self.p2, p3_array[j]
                self.clear_history_and_change_p(p1,p2,p3)
                I , varI = self.gather_and_measure_sample(t_equil , t_decorrel, sample_size)
                DATA_AS_ARRAY[i,j, 0] = I
                DATA_AS_ARRAY[i,j, 1] = varI

        np.savez(filename , p1 = p1_array , p3 = p3_array, I_N = DATA_AS_ARRAY[:,:, 0], varI_N = DATA_AS_ARRAY[:,:, 1])
        data = np.flip(DATA_AS_ARRAY[:,:,0].reshape((len1,len3)) , axis = 0)
        fig, ax = plt.subplots()
        im = plt.imshow(data, extent = [0,1,0,1])
        fig.colorbar(im , label = r'$<I>$/N', ax = ax , fraction=0.045)
        plt.savefig('preliminary_pd')
        plt.show()


    def visualize(self, n_sweeps = 1000):
        fig, ax = plt.subplots()
        #ax = plt.axes(xlim=(0, self.size-1), ylim=(0, self.size-1))
        image  = ax.imshow(self.lattice, cmap = 'inferno' , vmin = 0, vmax = 3 )

        for i in range(n_sweeps):
            self.update()
            image.set_array(self.lattice)
            plt.pause(0.001)
        plt.show()
    


    # def Immune_averageI_graph(self, p1, p2, p3, Immunerange = [0,1], resolution = 0.05 ):
    #     if Immunerange[1] <= Immunerange[0]:
    #         raise Exception(f'You are providing an unsuitable range.Immunerange = {Immunerange} . It should be [min,max]' )
    #     n_points = int((Immunerange[1] - Immunerange[0])/resolution)
    #     Immunities = np.linspace(Immunerange[0] ,Immunerange[1], n_points)
    #     Infections = np.zeros_like(Immunities)
    #     Infection_err = np.zeros_like(Immunities)

    #     for i in tqdm(range(n_points)):
    #         immunity = Immunities[i]
    #         self.clear_history_and_change_p(p1, p2, p3, new_permimmune = immunity)
    #         meanI, varI = self.gather_and_measure_sample( t_equil=100 , t_decorrel=1, sample_size = 100)
    #         Infections[i]= meanI
    #         Infection_err[i] = varI

    #     Infection_err = np.sqrt(Infection_err)
    #     fig, ax = plt.subplots()
    #     ax.errorbar(Immunities, Infections, Infection_err, capsize = 3, ls = 'none', marker = 'x')
    #     ax.set_xlabel(r'Immune fraction')
    #     ax.set_ylabel(r'$<f_I>$')
    #     plt.show()
    #     fig.savefig('permanent immunity effects.png')
    #     np.savez('effect_of_permanent_immunity',Immunities = Immunities, Infections = Infections, Infection_err = Infection_err)

    # def variance_withp1_graph(self, collect_data = True):

    #     if collect_data == False:
    #         data_arrays = np.load('Variance_graph.npz')
    #         p1 = data_arrays['p1']
    #         varI_N = data_arrays['varI_N'].reshape(p1.size)
    #         varI_N_err = data_arrays['varI_N_err'].reshape(p1.size)
    #         fig, ax = plt.subplots()
    #         ax.errorbar(p1, varI_N, varI_N_err)
    #         ax.set_xlim(0,1)
    #         ax.set_xlabel(r'$p_1$')
    #         ax.set_ylabel(r'$var(I/N)$')
    #         plt.show()
    #         fig.savefig('SIRS variance over p1.png')

#MULT. VARIANCE TO GET EXTENSIVE VAS. (Cv)
            


# plt.errorbar(p1, I_N, np.sqrt(varI_N))



SIRS_model = SIRS(50, 0.8 ,0.5,0.8)
#SIRS_model.count_infected()
#SIRS_model.visualize()
SIRS_model.p1p3_phase_diagram( resolution = 0.2)
#SIRS_model.plot_p1_p3_plase_diagram(collect_data = False )