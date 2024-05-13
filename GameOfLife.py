import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import random
from scipy.signal import convolve2d
import matplotlib.animation as animation

class GameOfLife():
    
    """Class to call on the game of life"""
    def __init__(self , size, initial_pattern = 'random'):
        """size, initial_pattern"""
        self.size = size
        self.initial_conditions = initial_pattern

        
        # initialize the lattice
        if initial_pattern == 'random':
            self.lattice = np.random.choice([0,1], size= (self.size , self.size))
        elif initial_pattern == 'blinker':
            lattice = np.zeros((self.size , self.size))
            point = (0,0)
            x,y = point[0], point[1]
            self.pattern = np.array([[0,1,0],[0,1,0],[0,1,0] ])
            lattice[x%self.size: (x+3)%self.size, y%self.size: (y+3)%self.size] = self.pattern
            self.lattice = np.copy(lattice) 
        elif initial_pattern == 'glider':
            lattice = np.zeros((self.size , self.size))
            #point = np.random.randint(0, self.size, size = 2)
            point = (40,40)
            x , y = point[0] , point[1]
            self.pattern = np.array([[1,1,0],[1,0,1],[1,0,0] ])
            lattice[x: (x+3)%self.size, y: (y+3)%self.size] = self.pattern
            self.lattice = np.copy(lattice) 
        else:
            raise Exception("Your initial pattern hasn't been defined--- define and try again")
            

    def NN_count_matrix(self):
        """given point tuple of indices (i,j) , 
        returns the sum of the neighbours of that point in self.lattice"""
        convolving_matrix = np.array([[1,1,1], [1,0,1], [1,1,1]])
        NN_counted_matrix = convolve2d(self.lattice,convolving_matrix ,mode = 'same', boundary = 'wrap')
        return NN_counted_matrix

    def clear_history(self):
        self.__init__(self.size, self.initial_conditions)
        
    
    def update(self, n_sweeps = 1):
        """ updates self.lattice deterministically, in parallel"""
        if self.count_livesites(update = False) != 0:
            # make a new lattice (it starts with all points dead)
            new_lattice = np.zeros_like(self.lattice)

            # update the lattice
            for _ in range(n_sweeps):
                # for every point in the lattice
                NN_counts = self.NN_count_matrix()
                live_to_live_condition1 = (self.lattice == 1)&(NN_counts ==2)
                live_to_live_condition2 = (self.lattice == 1)&(NN_counts ==3)
                new_lattice[live_to_live_condition1] = 1
                new_lattice[live_to_live_condition2] = 1
                dead_to_live_condition = (self.lattice == 0)&(NN_counts ==3)
                new_lattice[dead_to_live_condition] = 1
                #update self.lattice every sweep
                self.lattice = new_lattice
    
    def comass(self):
        """does what it says in the tin. returns an array with x-com , y-com"""
        args = np.argwhere(self.lattice ==1)
        comass = np.sum(args, axis = 0) / self.lattice[self.lattice ==1].size
        return comass
    
    def track_comass(self, n_sweeps = 50):
        """does what it says in the tin. returns an array with x-com , y-com, t-com"""
        comxlist , comylist, steplist = [] , [] , []
        comx , comy = self.comass()
        for step in range(n_sweeps):
            self.update()
            comxnew, comynew = self.comass()
            if np.abs(comxnew - comx) < 4:
                if np.abs(comynew - comy) < 4:
                    comx , comy = comxnew , comynew
                    comxlist.append(comxnew) , comylist.append(comynew), steplist.append(step)
        return np.array([comxlist, comylist, steplist])
    

    def visualize_track_comass(self,  n_sweeps = 50, filename = 'GOL_tracked_comass', figname ='GOL_comass_figure' ):
        """rund track_comass and smakes a plot"""
        com_history = self.track_comass(n_sweeps = n_sweeps)
        fig, ax = plt.subplots()
        x = com_history[0,:]
        y = com_history[1,:]
        steps = com_history[2,:]
        ax.scatter(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        fig.savefig(figname)
        np.savez(filename,comx_history =com_history[0,:] , comy_history = com_history[1,:], t_history = com_history[2,:])
        x = com_history[0,:]
        x_vel = (x[-1] - x[0])/(steps[-1] - steps[0])
        y_vel = (y[-1] - y[0])/(steps[-1] - steps[0])
        vel = np.sqrt(x_vel**2  + y_vel**2)
        print(f' v_x = {x_vel} , v_y = {y_vel}. v = {vel} ')

    
    def count_livesites(self, update = True ):
        n_live = np.sum(self.lattice)
        if update == True:
            self.update()      
        return n_live
    
    def count_equilibration_lifetime(self, n_sweeps_threshold = 5000):
        #count_array = np.zeros(n_sweeps_threshold)
        duplicates = 0
        for i in range(n_sweeps_threshold):
            
            old_livesites = self.count_livesites(update = True)
            new_livesites = self.count_livesites(update = False)
            #count_array[i] = new_livesites
            
            if old_livesites == new_livesites: duplicates +=1

            if duplicates > 100: return i-100
        return np.inf
    

    def equilibration_lifetime_experiment(self, n_samples = 1000, n_sweeps_threshold = 3000, save_data = True):
        
        equilibration_lifetimes = np.zeros(n_samples, dtype = float)
        for i in tqdm.tqdm(range(n_samples)):
            lifetime = self.count_equilibration_lifetime(n_sweeps_threshold = n_sweeps_threshold)
            equilibration_lifetimes[i]=lifetime
            # clear history
            self.clear_history()
        
        equilibration_lifetimes = equilibration_lifetimes[np.isfinite(equilibration_lifetimes)]
        fig, ax = plt.subplots()
        ax.hist(equilibration_lifetimes , bins = 50)
        ax.set_xlabel('Equilibration time /sweeps')
        ax.set_ylabel('Counts')
        ax.set_xlim(0, n_sweeps_threshold)
        plt.show()
        fig.savefig('GAME_OF_LIFE_EQUILIBRATION_TIME')

        if save_data == True:
            np.save('GAME_OF_LIFE_EQUILIBRATION_TIME', equilibration_lifetimes )
        return equilibration_lifetimes
        

GOL = GameOfLife(50, initial_pattern='random')
#GOL.visualize_track_comass()
GOL.equilibration_lifetime_experiment(save_data=True)