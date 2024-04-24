import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import random
from scipy.signal import convolve2d
import matplotlib.animation as animation


class Ising2D():

    def __init__(self, size=50, T=2,  start='random'):
        self.start = start
        self.size = size
        self.T = T
        self.sweep = size**2


        ### Make the lattices for different starting cases
        if start == 'random':
            self.lattice = np.random.choice([-1,1], size = (self.size, self.size))
        elif start == 'ground':
            self.lattice = np.ones((self.size, self.size), dtype = int)
        elif start == 'half':
            self.lattice = np.ones((self.size, self.size), dtype = int)
            self.lattice[int(self.size/2): , :] = 0


    ### reinitializers
    def reinitialize(self):
        """Clear history"""
        self.__init__(self.size, self.T, start = self.start)

    def reinitialize_changevars(self, Tnew, start = 'original'):
        """Clear history and change T"""
        if start == 'original':
            self.__init__(self.size, Tnew, start = self.start)
        else:
            self.__init__(self.size, Tnew, start = start)
        

    ### Sum of nn spins        
    def NN_spins_sum(self, i,j):
        """Compute the hamiltonian for a single spin s_ij and its neighbours"""
        top = (i-1)%self.size
        bottom = (i+1)%self.size
        left = (j-1)%self.size
        right = (j+1)%self.size

        S = self.lattice[top, j] + self.lattice[bottom, j] + self.lattice[i, left] + self.lattice[i, right]
        return S
    

    
    def glauber(self, nsweeps=1):
        """you guessed it!"""
        # N = nsweeps*self.sweep
        # #make a single array holding the indices of the spins we'll be attempting to update
        # indices = np.random.randint(0, self.size, size = (N, 2))
        # for run in range(N):
        #     #select the point we'll be working with
        #     row, column =indices[run,0] , indices[run,1]
        #     E_before = -self.lattice[row, column]*self.NN_spins_sum(row, column)
        #     E_after = -E_before
        #     DE = E_after - E_before
        #      # find probability of switching
        #     P = np.min(np.array([1, np.exp(-DE/self.T)]))
        #     self.lattice[row, column] *= random.choices([-1, 1], weights = [P, 1-P])[0]
        
        """you guessed it!"""
        N = nsweeps*self.sweep
        #make a single array holding the indices of the spins we'll be attempting to update
        
        for _ in range(nsweeps):
            indices = np.random.randint(0, self.size, size = (self.sweep, 2))
            for run in range(self.sweep):
                #select the point we'll be working with
                row, column =indices[run,0] , indices[run,1]
                E_before = -self.lattice[row, column]*self.NN_spins_sum(row, column)
                E_after = -E_before
                DE = E_after - E_before
                # find probability of switching
                P = np.min(np.array([1, np.exp(-DE/self.T)]))
                self.lattice[row, column] *= random.choices([-1, 1], weights = [P, 1-P])[0]

        # for run in range(N):
        


        #     #select the point we'll be working with
        #     row, column =random.randint(0, self.size-1) , random.randint(0, self.size-1)
        #     E_before = -self.lattice[row, column]*self.NN_spins_sum(row, column)
        #     E_after = -E_before
        #     DE = E_after - E_before
        #     # find probability of switching
        #     P = np.min(np.array([1, np.exp(-DE/self.T)]))
        #     self.lattice[row, column] *= random.choices([-1, 1], weights = [P, 1-P])[0]



    def kawasaki(self, nsweeps = 1):
        """you guessed it my bro"""
        N = nsweeps*self.sweep
        #make a single array holding the indices of the spins we'll be attempting to update
        indices = np.random.randint(0, self.size, size = (N, 4))
        decisions = np.random.random(size = N)
        for run in range(N):
            #set the points to work with
            row1 = indices[run,0]
            column1 =  indices[run,1]
            row2 = indices[run, 2]
            column2 = indices[run, 3]
            #find the spins and nearest neighbours of both spins
            S1 = self.NN_spins_sum(row1, column1)
            spin1 = self.lattice[row1, column1]
            S2 = self.NN_spins_sum(row2, column2)
            spin2 = self.lattice[row2, column2]
            # deal with the case where spins 1 and 2 are nearest neighbours
            if np.abs(row1%self.size - row2%self.size ) == 1 and np.abs(column1%self.size - column2%self.size ) == 1:
                S1 +=  - spin2
                S2 +=  - spin1
            #look at change in energy associated with swapping places
            E_old = -S1*spin1  - S2*spin2
            E_new = -S1*spin2  - S2*spin1
            DE = E_new - E_old
            # find probability of swapping places
            P = np.min(np.array([1, np.exp(-DE/self.T)]))
            if decisions[run] <= P:
                self.lattice[row1, column1] = spin2
                self.lattice[row2, column2] = spin1

    def magnetization(self):
        """returns the total magnetization """
        return np.sum(self.lattice)

    def energy (self):
        """returns the total energy of the stystem"""
        NN_to_consider = np.array([[0,1,0], [1,0,1], [0,1,0]])
        E_in_sites = - 0.5*convolve2d(self.lattice, NN_to_consider, mode = 'same', boundary = 'wrap')*self.lattice
        E = np.sum(E_in_sites)
        return E
    
    def visualize_glauber(self):

        self.reinitialize()
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.size-1), ylim=(0, self.size-1))
        image  = ax.imshow(self.lattice)
        # animation function.  This is called sequentially
        def animate(i):
            self.glauber(nsweeps=1)
            image.set_data(self.lattice)
            return [image]
        #make the animation
        anim = animation.FuncAnimation(fig, animate,
                                frames=200, interval=25)
        plt.show()

    
    def visualize_kawasaki(self):
        self.reinitialize_changevars(Tnew=self.T, start = 'half')
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.size-1), ylim=(0, self.size-1))
        image  = ax.imshow(self.lattice)
        # animation function.  This is called sequentially
        def animate(i):
            self.kawasaki(nsweeps=1)
            image.set_data(self.lattice)
            return [image]
        #make the animation
        anim = animation.FuncAnimation(fig, animate,
                                frames=200, interval=25)
        plt.show()


    def bootstrap_variance(self, array, nsets = 1000, setsize = 'same'):
        if setsize == 'same': setsize = len(array)
        #sets = np.zeros((nsets, setsize))
        selected_indices = np.random.randint(0 , len(array), size = (setsize,nsets))
        sets = array[selected_indices]
        variance = np.var(sets, axis = 0)
        errors = np.std(variance, axis = 0)
        return errors


    def glauber_data_gathering( self, Tmin=1, Tmax=3, Ntemperatures = 20, t_equil = 100 , t_decorrel =10, sample_size = 100):
        # make arrays to store data (intermediate storage: will use pd later)
        T_array = np.linspace(Tmin,Tmax, Ntemperatures)
        M_array = np.zeros(sample_size)
        E_array = np.zeros(sample_size)
        #and the dictionary for the final DF:
        data = {"T":[], "M":[],"M_std":[], "E":[], "E_std": [], "m":[], "m_std":[], "chi_m":[], "chi_m_std":[], "c":[], "c_std":[], "Nspins":[]}

        # gather and save data for every T
        for i in tqdm.tqdm(range(len(T_array))):
            #Select new T
            T = T_array[i]
            #reinitialize from ground state with the new T
            self.reinitialize_changevars(Tnew = T, start = 'ground')
            #run model for the equilibration time
            self.glauber(nsweeps=t_equil)

            #Now let's take one measurement every t_decorrel
            for j in tqdm.tqdm(range(sample_size)):
                M_array[j] = self.magnetization()
                E_array[j] = self.energy()
                self.glauber(nsweeps = t_decorrel)
            data['T'].append(T)
            data['M'].append(np.mean(M_array))
            data['M_std'].append(np.std(M_array))
            data['E'].append(np.mean(E_array))
            data['E_std'].append(np.std(E_array))
            data['m'].append(np.mean(M_array)/self.sweep)
            data['m_std'].append(np.std(M_array)/self.sweep)
            data['chi_m'].append(np.var(M_array)/(T*self.sweep))
            data['chi_m_std'].append(self.bootstrap_variance(M_array, nsets = sample_size)/(self.sweep*T))
            data['c'].append(np.var(E_array)/(self.sweep*T**2))
            data['c_std'].append(self.bootstrap_variance(E_array, nsets = sample_size )/(self.sweep*T**2))
            data['Nspins'].append(self.sweep)

        data = pd.DataFrame(data)
        data.to_csv('glauber.csv')

        fig, ax = plt.subplots(2,2, figsize = (8,8),sharex = True)
        figm = ax[0,0]
        figE = ax[0,1]
        figchi = ax[1,0]
        figc = ax[1,1]
        #figm.set_xlabel('T')
        figm.set_ylabel('|m|')
        figE.set_ylabel('E')
        figchi.set_ylabel(r'$\chi_m$')
        figc.set_ylabel(r'c per spin')
        figchi.set_xlabel('T')
        figc.set_xlabel('T')

        figm.errorbar(data['T'],data['m'], data['m_std'], capsize = 2, color = 'k', marker = 's')
        figchi.errorbar(data['T'],data['chi_m'], data['chi_m_std'], capsize = 2, color = 'k', marker = 's')
        figE.errorbar(data['T'],data['E'], data['E_std'], capsize = 2, color = 'k', marker = 's')
        figc.errorbar(data['T'],data['c'], data['c_std'], capsize = 2, color = 'k', marker = 's')
        plt.tight_layout()
        fig.savefig('ising_glauber_results.png')
        plt.show()

    def kawasaki_data_gathering( self, Tmin=0.1, Tmax=2, Ntemperatures = 20, t_equil = 100 , t_decorrel =10, sample_size = 100):
        # make arrays to store data (intermediate storage: will use pd later)
        T_array = np.linspace(Tmin,Tmax, Ntemperatures)
        E_array = np.zeros(sample_size)
        #and the dictionary for the final DF:
        data = {"T":[],  "E":[], "E_std": [],"c":[], "c_std":[], "Nspins":[]}

        # gather and save data for every T
        for i in tqdm.tqdm(range(len(T_array))):
            #Select new T
            T = T_array[i]
            #reinitialize from ground state with the new T
            self.reinitialize_changevars(Tnew = T, start = 'half')
            #run model for the equilibration time
            self.kawasaki(nsweeps=t_equil)

            #Now let's take one measurement every t_decorrel
            for j in tqdm.tqdm(range(sample_size)):
                E_array[j] = self.energy()
                self.kawasaki(nsweeps = t_decorrel)

            data['T'].append(T)
            data['E'].append(np.mean(E_array))
            data['E_std'].append(np.std(E_array))
            data['c'].append(np.var(E_array)/(self.sweep*T**2))
            data['c_std'].append(self.bootstrap_variance(E_array, nsets = sample_size )/(self.sweep*T**2))
            data['Nspins'].append(self.sweep)

        data = pd.DataFrame(data)
        data.to_csv('kawasaki.csv')

        fig, ax = plt.subplots(1,2, figsize = (8,4),sharex = True)
        figE = ax[0]
        figc = ax[1]
        #figm.set_xlabel('T')

        figE.set_ylabel('E')
        figc.set_ylabel(r'c per spin')
        figc.set_xlabel('T')
        figE.errorbar(data['T'],data['E'], data['E_std'], capsize = 2, color = 'k', marker = 's')
        figc.errorbar(data['T'],data['c'], data['c_std'], capsize = 2, color = 'k', marker = 's')
        plt.tight_layout()
        fig.savefig('ising_kawasaki_results.png')
        plt.show()
        


Ising = Ising2D(size = 50, T=1)
# Ising = Ising2D(size = 50, T=2, start = 'half')
# Ising.visualize_kawasaki()
Ising.visualize_glauber()

# Ising.glauber_data_gathering(Ntemperatures = 20)
# Ising.kawasaki_data_gathering(Ntemperatures = 20)

            






        
            





    




    






