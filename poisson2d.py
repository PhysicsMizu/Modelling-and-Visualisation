import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from tqdm import tqdm 
import time
from scipy.signal import convolve2d as convolve2d
from scipy.ndimage import convolve as convolve
from scipy.optimize import curve_fit

class Magnetic_problem():
    
    """Class to call on 2D Poisson (magnetic) with square lattice and pbcs"""
    def __init__(self , size = 50, current_dist = 'wire', errortol = 1e-03):
        
        self.size = size
        self.charge_dist = current_dist
        self.ndims = 2
        self.error_tolerance = errortol

        if  current_dist == 'wire':
            rho = np.zeros((self.size , self.size))
            rho[int(self.size/2),int(self.size/2)] = 1.
            self.rho = rho
            self.potential = np.zeros( shape= (self.size , self.size))
        else:
            raise Exception("Your charge/current density lattice hasn't been defined--- define and try again")


    def clear_history(self):
        self.__init__( size = self.size, charge_dist = self.charge_dist)

    
    def distancefromcenter(self, array):
        "you guessed it!"
        x0 , y0 = int(array.shape[0]/2) , int(array.shape[1]/2)
        x, y = np.indices(array.shape)
        x -= x0
        y -= y0
        distance_from_center = np.sqrt(x**2 + y**2)
        return distance_from_center
    

    #     y_nn = np.array([[0,-1,0] , [0,0,0], [0,1,0] ])
    #     partial_y = convolve2d(field, y_nn, mode = 'same', boundary = 'wrap') / (2*self.dx)
    #     return np.array([partial_x, partial_y])


    def jacobi_update(self, nruns = 1):
        #uses convolutions to do jacobi efficiently

        nn = np.array( [[0,1,0] , [1,0,1], [0,1,0] ] )
        for n in tqdm(range(nruns)):
            oldphi = np.copy(self.potential)
            laplacian_terms = convolve(self.potential, nn, mode ='constant', cval = 0 )
            newphi = (laplacian_terms + self.rho)/(2*self.ndims)
            self.potential = newphi
            error = np.sum(np.abs(newphi - oldphi))
            if error < self.error_tolerance:
                return n
        

    def gauss_seidel_update(self, nruns = 1, omega = 1):
        """the swaggiest code everrrrr - checkerboards for gauss-seidel muahahahahaaaa
        omega = 1 corresponds to GS"""
       
        checkerboard = np.indices((self.size, self.size)).sum(axis=0) % 2
        nn = np.array( [[0,1,0] , [1,0,1], [0,1,0] ] )

        for n in range(nruns):
            phi_old =  np.copy(self.potential)
            # update 1s first
            self.potential[checkerboard==1] = ((convolve( self.potential, nn, mode ='constant', cval = 0 ) + self.rho)/(2*self.ndims))[checkerboard==1]
            self.potential = omega*self.potential + (1-omega)*phi_old
            phi_old = np.copy(self.potential)
            #now update the white squares in the checkerboard. overrelaxation already included. GS
            self.potential[checkerboard==0]= ( (convolve(self.potential, nn, mode ='constant', cval = 0 ) + self.rho)/(2*self.ndims))[checkerboard==0]
            self.potential = omega*self.potential + (1-omega)*phi_old
            #check convergence
            if np.sum(np.abs(self.potential - phi_old ))   <= self.error_tolerance: return n


    def GS_equilibration_andvisualize(self, runslim = 20000):
        numequil = self.gauss_seidel_update(nruns = runslim)
        self.plotE_andPhi_v_d('GaussSeidel50')

    
    def OR_equilibration_andvisualize(self,omega = 1.9 ,runslim = 20000):
        numequil = self.gauss_seidel_update(nruns = runslim, omega = omega)
        self.plotE_andPhi_v_d('OverRelaxation50')


    def B_field(self):
        if self.ndims !=2: raise Exception("HOLD YOUR HORSES..... I am a simulation made to model magnetic problems with cylindrical symmetry for infinitely long j")

        """ the field is a 3D array. Using 4nn... Using PBCs"""
        x_nn = np.array([[0,0,0] ,
                         [-1,0,1], 
                         [0,0,0] ])
        partial_x = convolve(self.potential, x_nn, mode = 'constant', cval = 0)/2

        y_nn = np.array([[0,-1,0],
                         [0,0,0],
                         [0,1,0] ])
        partial_y = convolve(self.potential, y_nn, mode = 'constant', cval = 0)/2

        B = np.array([partial_y, -partial_x])
        B[:,int(self.size/2) ,int(self.size/2) ] = 0
        return B

    
    def magnetic_problem(self):
        """Runs the PDE solver and plots the potential, field_strength snd the field direction"""
        if self.ndims !=2:
            raise Exception("I am made for magnetic problems with no z dependence")
        # IN 2D, j = rho
        #A is along z , so we go back to the poisson potential in 2D
        
        #solve thr PDE
        self.gauss_seidel_update( omega = 1.7, nruns = 3000)
        # make the x, y arrays
        x = np.linspace(0, self.size -1,  self.size)
        y = np.linspace(0, self.size -1,  self.size)
        X, Y = np.meshgrid(x,y)
        #cal
        B = self.B_field()
        fig, ax = plt.subplots(1,3, figsize = (16, 5))
        figA = ax[0]
        figB = ax[1]
        figBdir = ax[2]
        plotA = figA.contour(X, Y, self.potential, levels = 10)
        figA.set_xlim(0,50)
        figA.set_ylim(0,50)
        figA.clabel(plotA, inline=True, fontsize=10)
        figA.set_aspect('equal')
        figA.set_title(r'$A_z$')

        Bstrength = np.sqrt(np.sum(B**2, axis = 0))
        plotB = figB.contour(X, Y, Bstrength, levels = 10)
        figB.set_xlim(0,50)
        figB.set_ylim(0,50)
        figB.clabel(plotB, inline=True, fontsize=10)
        figB.set_aspect('equal')
        figB.set_title('B strength')

        #quiver plot
        B2quiver = B/np.sqrt(np.sum(B**2, axis =0))
        Bx2quiver = B2quiver[0,:,:]
        By2quiver = B2quiver[1,:,:]
        figBdir.imshow(Bstrength)
        figBdir.quiver(X,Y, Bx2quiver, By2quiver, color = 'r', angles='xy', scale_units='xy', scale=1, width = 0.01)
        figBdir.set_xlim(20,30)
        figBdir.set_ylim(20,30)
        figBdir.set_aspect('equal')
        figBdir.set_title('B direction')

        figA.set_xlabel(r' $x/\delta x$')
        figB.set_xlabel(r' $x/\delta x$')
        figBdir.set_xlabel(r' $x/\delta x$')
        figA.set_ylabel(r' $y/\delta x$')
        figB.set_ylabel(r' $y/\delta x$')
        figBdir.set_ylabel(r' $y/\delta x$')

        fig.savefig('magnetic problem contours.png', dpi = 500)
        plt.show()

        self.plotB_andA_v_d(filename = 'Magnetic' )

        






    def plotB_andA_v_d(self, filename= 'Magnetic'):

        d = self.distancefromcenter(self.potential)
        A = self.potential
        B = self.B_field()
        Bstrength = np.sqrt(np.sum(B**2, axis = 0))

       

        dict = {'x':[], 'y':[],'d':[] ,'A':[], 'B':[], 'Bx':[], 'By':[]}
        for i in range(self.size):
            for j in range(self.size):
                dict['x'].append(i)
                dict['y'].append(j)
                dict['d'].append(d[i,j])
                dict['A'].append(A[i,j])
                dict['B'].append(Bstrength[i,j])
                dict['Bx'].append(B[0,i,j])
                dict['By'].append(B[1,i,j])


        data_tosave = pd.DataFrame(dict)
        data_tosave.to_csv(filename)
        fig , ax = plt.subplots(2,1, figsize = (8,12), sharex = True)
        figA = ax[0]
        figB = ax[1]

        dorder = np.argsort(d.flatten())
        dord = d.flatten()[dorder]
        Aord = A.flatten()[dorder]
        Bstrengthord = Bstrength.flatten()[dorder]
        def power(x,a,n):
            return a*x**n
        condition = (dord < 10)&(dord >1)
        Apopt, Apcov = curve_fit(power, dord[condition], Aord[condition])
        Bpopt, Bpcov = curve_fit(power, dord[condition], Bstrengthord[condition])

        figA.loglog(dord,Aord, linestyle = '', marker = 'x', color = 'k')
        figA.loglog(dord, power(dord, Apopt[0], Apopt[1]), linestyle = '--', color = 'b', label = f'exponent = {np.round(Apopt[1], decimals = 1)}')
        figB.loglog(dord,Bstrengthord, linestyle = '', marker = 'x', color = 'k')
        figB.loglog(dord,power(dord, Bpopt[0], Bpopt[1]), linestyle = '--', color = 'b', label = f'exponent = {np.round(Bpopt[1], decimals = 1)}')
        figB.legend(loc = 'lower left')
        figA.legend(loc = 'lower left')
        figB.set_xlabel('d')
        figA.set_ylabel(r'$A$')
        figB.set_ylabel(r'$B$')

        plt.tight_layout()
        fig.savefig(filename)
        plt.show()
        
  







    
        

#PS = Magnetic_problem(size =50)
#PS.jacobi_equilibration_andvisualize()
# PS.clear_history()
# PS.GS_equilibration_andvisualize()
# PS.clear_history()
# PS.OR_equilibration_andvisualize(runslim = 20000)
# PS.clear_history() 
#PS.OR_explore_omegas()           
MP = Magnetic_problem(size= 50 , current_dist = 'wire')
MP.magnetic_problem()