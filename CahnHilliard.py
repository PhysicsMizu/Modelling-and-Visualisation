import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from tqdm import tqdm 
import time
from scipy.signal import convolve2d as convolve2d

class CahnHilliard():
    
    """Class to call on Cahn Hilliard with square lattice and pbcs"""
    def __init__(self , size, a =0.1, k=0.1,M = 0.1, dx = 1, dt = 1, phi0 = 0.5 , initial_pattern = 'random'):
        
        self.size = size
        self.initial_pattern = initial_pattern
        self.dx = dx
        self.dt = dt
        self.a = a
        self.k = k
        self.M =M
        self.phi0 = phi0

           
        # self.sweep = self.size**2

        if initial_pattern == 'random':
            self.concentration = self.phi0*np.ones((self.size , self.size)) + 0.05*np.random.uniform(low = -1. , high = 1, size= (self.size , self.size))
            self.mu = self.chemical_potential()
            self.f = self.free_energy_density()
        else: raise Exception("Your initial pattern hasn't been defined--- define and try again")
            


    def clear_history(self):
        """Clears history by reinitializing the lattice but keeps the original lattice size and variables"""
        self.__init__(self.size, self.a, self.k, self.M, self.dx, self.dt, self.phi0, self.initial_pattern)

        
    def clear_history_and_change_vars(self, newa   , newk , newM, newdx, newdt, newphi0, newinitial_pattern ):
        """Clears history by reinitializing the lattice. also changes supplied variable (these should be floats)"""
        self.__init__(self.size, newa, newk, newM, newdx, newdt, newphi0, newinitial_pattern)
        
    def clear_history_and_change_phi0(self, newphi0):
        """Clears history by reinitializing the lattice. also changes supplied variable (these should be floats)"""
        self.__init__(self.size, self.a, self.k, self.M, self.dx, self.dt, newphi0, self.initial_pattern)


 

    def discrete_laplacian(self, field):
        """ the field is a 2D array. Using 4nn... Using PBCs"""
        nn = np.array([[0,1,0] , [1,0,1], [0,1,0] ])
        laplacian = (convolve2d(field, nn, mode = 'same', boundary = 'wrap') - 4*field)/(self.dx**2)
        return laplacian
        

    def discrete_grad(self, field):
        """ the field is a 2D array. Using 4nn... Using PBCs"""
        x_nn = np.array([[0,0,0] , [-1,0,1], [0,0,0] ])
        partial_x = convolve2d(field, x_nn, mode = 'same', boundary = 'wrap') / (2*self.dx)
        y_nn = np.array([[0,-1,0] , [0,0,0], [0,1,0] ])
        partial_y = convolve2d(field, y_nn, mode = 'same', boundary = 'wrap') / (2*self.dx)
        return np.array([partial_x, partial_y])


    def discrete_gradsquared(self,field):
        """ the field is a 2D array. Using 4nn... Using PBCs"""
        grad = self.discrete_grad(field)
        gradsquared = np.sum(grad**2 , axis = 0)
        return gradsquared

    def chemical_potential(self):
        mu = -self.a*self.concentration + self.a*self.concentration**3 - self.k*self.discrete_laplacian(self.concentration)
        self.mu = mu
        return mu

    def free_energy_density(self):
        f = -(self.a /2)*self.concentration**2 + (self.a/4)*self.concentration**4 + (self.k/2)*self.discrete_gradsquared(self.concentration)
        self.f = f
        return f
    
    def free_energy(self):
        return np.sum(self.f*self.dx**2)

    def update(self, nruns = 1):
        #nruns = int(time/self.dt)
        for i in range(nruns):
            new_conc = self.concentration + self.M*self.dt*self.discrete_laplacian(self.chemical_potential())
            self.concentration = new_conc

    def visualize(self, n_sweeps = 10000):

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.size-1), ylim=(0, self.size-1))
        image  = ax.imshow(self.concentration, cmap = 'gnuplot', vmin = 0, vmax = 1)
        fig.colorbar(image , ax = ax, label = r'$\phi$')


        # animation function.  This is called sequentially
        def animate(i):
            self.update(100)
            image.set_data(self.concentration)
            return [image]
        #make the animation
        anim = animation.FuncAnimation(fig, animate,
                            frames=n_sweeps, interval=1)
        
        plt.show()
        #plt.pause(1)import numpy as np




    
    def track_f(self, runs = 100000):
        data = {'Free Energy':[] , 'runs':[]}
        
        for run in tqdm(range(runs)):
            data['Free Energy'].append(self.free_energy())
            data['runs'].append(run)
            self.update()
            # if ((np.abs(np.array(data['Free Energy'][-1000:-2]) - data['Free Energy'][-1]).all() < 1e03)and(run > 2000))== True:
            #     break
        return pd.DataFrame(data)
        #plt.plot(data['runs'], data['Free Energy'])
        #plt.show()

    def track_f_for_phi0_range(self, phi0_iterable= [0, 0.5], runs = 50000):
        """given an iterable phi0 (list of concentrations), stores free energy 'runs' sweeps"""
        fig, ax = plt.subplots()
        for phi0 in tqdm(phi0_iterable):
            self.clear_history_and_change_phi0(newphi0= phi0)
            data = self.track_f(runs=runs)
            label = r'$\phi_0$ =' + str(phi0)
            ax.plot(data['runs'], data['Free Energy'], label = label)
            data['phi0'] = phi0
            title = 'phi0' + str(phi0) + 'free_energy'
            data.to_csv(title)

        ax.legend()
        ax.set_xlabel('sweeps')
        ax.set_ylabel('F')
        plt.show()
        fig.savefig('F v runs.png', dpi = 300)

            
CH = CahnHilliard(50, phi0=0.5)
CH.visualize()
CH.track_f_for_phi0_range(phi0_iterable=[0 ,0.5], runs = 100000)
#CH.track_f()




    




