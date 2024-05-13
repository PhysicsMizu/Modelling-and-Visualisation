import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from tqdm import tqdm 
from scipy.signal import convolve2d as convolve2d
from scipy.ndimage import convolve as convolve
from scipy.optimize import curve_fit

class Poisson():
    
    """Class to call on Poisson with square lattice and dirichlet boundary conditions"""
    def __init__(self , size = 50, charge_dist = 'monopole', errortol = 1e-03):
        
        self.size = size
        self.charge_dist = charge_dist
        self.error_tolerance = errortol
        self.ndims = 3

        if  charge_dist == 'monopole':
            rho = np.zeros((self.size , self.size, self.size))
            rho[int(self.size/2),int(self.size/2), int(self.size/2)] = 1.
            self.rho = rho
            self.potential = np.zeros( shape= (self.size , self.size, self.size))
            


    def clear_history(self):
        self.__init__( size = self.size, charge_dist = self.charge_dist)

    def E_field(self):
        """ the field is a 3D array. Using 4nn... Using PBCs"""
        x_nn = np.array([[[0,0,0] , [0,0,0], [0,0,0] ],\
                        [[0,0,0] , [-1,0,1], [0,0,0] ], \
                        [[0,0,0] , [0,0,0], [0,0,0] ]])
        partial_x = convolve(self.potential, x_nn, mode = 'constant', cval = 0)/2

        y_nn = np.array([[[0,0,0] , [0,0,0], [0,0,0] ],\
                        [[0,-1,0] , [0,0,0], [0,1,0] ], \
                        [[0,0,0] , [0,0,0], [0,0,0] ]])
        partial_y = convolve(self.potential, y_nn, mode = 'constant', cval = 0)/2

        z_nn = np.array([[[0,0,0] , [0,-1,0], [0,0,0] ],\
                        [[0,0,0] , [0,0,0], [0,0,0] ], \
                        [[0,0,0] , [0,1,0], [0,0,0] ]])
        partial_z = convolve(self.potential, z_nn, mode = 'constant', cval = 0)/2

        E = -np.array([partial_x, partial_y, partial_z])
        return E
    
    def distancefromcenter(self, array):
        "returns the distance from the center of an array"
        if self.ndims ==3:
            x0 , y0, z0  = int(array.shape[0]/2) , int(array.shape[1]/2) , int(array.shape[2]/2)
            x, y, z  = np.indices(array.shape)
            x -= x0
            y -= y0
            z -= z0
            distance_from_center = np.sqrt(x**2 + y**2 + z**2)
        return distance_from_center
    
    

    def plotE_andPhi_v_d(self, filename):
        '''Big, fat plotter'''
        d = self.distancefromcenter(self.potential)
        phi = self.potential
        E = self.E_field()
        Estrength = np.sqrt(np.sum(E**2, axis = 0))

       
        #measure everything
        dict = {'x':[], 'y':[], 'z':[],'d':[] ,'Phi':[], 'Estrength':[], 'Ex':[], 'Ey':[], 'Ez':[]}
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    dict['x'].append(i)
                    dict['y'].append(j)
                    dict['z'].append(k)
                    dict['d'].append(d[i,j,k])
                    dict['Phi'].append(phi[i,j,k])
                    dict['Estrength'].append(Estrength[i,j,k])
                    dict['Ex'].append(E[0,i,j,k])
                    dict['Ey'].append(E[1,i,j,k])
                    dict['Ez'].append(E[2,i,j,k])


        data_tosave = pd.DataFrame(dict)
        data_tosave.to_csv(filename)
        fig , ax = plt.subplots(2,1, figsize = (6,12), sharex = True)
        figphi = ax[0]
        figE = ax[1]

        dorder = np.argsort(d.flatten())
        dord = d.flatten()[dorder]
        phiord = phi.flatten()[dorder]
        Estrengthord = Estrength.flatten()[dorder]

        #do the fitting
        def power(x,a,n):
            return a*x**n
        condition = (dord < 10)&(dord >1)
        phipopt, phipcov = curve_fit(power, dord[condition], phiord[condition])
        Epopt, Epcov = curve_fit(power, dord[condition], Estrengthord[condition])

        #do the plotting
        figphi.loglog(dord,phiord, linestyle = '', marker = 'x', color = 'k')
        figphi.loglog(dord, power(dord, phipopt[0], phipopt[1]), linestyle = '--', color = 'b', label = f'exponent = {np.round(phipopt[1], decimals = 1)}')
        figE.loglog(dord,Estrengthord, linestyle = '', marker = 'x', color = 'k')
        figE.loglog(dord,power(dord, Epopt[0], Epopt[1]), linestyle = '--', color = 'b', label = f'exponent = {np.round(Epopt[1], decimals = 1)}')
        #format plots
        figE.legend(loc = 'lower left')
        figphi.legend(loc = 'lower left')
        figE.set_xlabel('d')
        figphi.set_ylabel(r'$\Phi$')
        figE.set_ylabel(r'$E$')
        plt.tight_layout()
        plt.show()
        fig.savefig(filename)


    def jacobi_update(self, nruns = 1):
        """ Updates and returns the number of updates needed for convergence"""
        nn = np.array([[[0,0,0] , [0,1,0], [0,0,0] ],\
                           [[0,1,0] , [1,0,1], [0,1,0] ],\
                           [[0,0,0] , [0,1,0], [0,0,0] ]] )
        for n in tqdm(range(nruns)):
            oldphi = np.copy(self.potential)
            
            laplacian_terms = convolve(self.potential, nn, mode ='constant', cval = 0 )
            newphi = (laplacian_terms + self.rho)/(2*self.ndims)
            self.potential = newphi
            error = np.sum(np.abs(newphi - oldphi))
            if error < self.error_tolerance:
                return n
        

    def jacobi_equilibration_andvisualize(self, runslim = 20000):
        """Makes all the plots you might want..."""
        #do the updating
        numequil = self.jacobi_update(nruns = runslim)
        #set a figure with the plots for E vs. d
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        figphi = ax[0]
        figE = ax[1]

        #plot the potential and E strength
        plot0 =figphi.imshow(self.potential[int(self.size/2), :,:])
        figphi.set_xlabel(r'$x/\delta x$')
        figphi.set_ylabel(r'$y/\delta x$')
        plot1 = figE.imshow(np.sqrt(np.sum(self.E_field()**2, axis = 0))[int(self.size/2),:,:])
        figE.set_xlabel(r'$x/\delta x$')
        figE.set_ylabel(r'$y/\delta x$')
        plt.colorbar(plot0, ax = figphi, label = r'$\phi$')
        plt.colorbar(plot1, ax = figE, label = r'E')

        #make the quiver plot
        E = self.E_field()[:, :,:,int(self.size/2)]
        print(E.shape)
        E2quiver = E/np.sqrt(np.sum(E**2, axis =0))
        Ex2quiver = E2quiver[1,:,:]
        Ey2quiver = E2quiver[2,:,:]
        print(Ex2quiver.shape)
        x = np.arange(self.size)
        y = np.arange(self.size)
        X,Y = np.meshgrid(x,y)
        figE.quiver(X,Y, -Ex2quiver, -Ey2quiver, color = 'r', angles='xy', scale_units='xy', scale=1, width = 0.01)
        
        #plot formatting
        figE.set_xlim(20,30)
        figE.set_ylim(20,30)
        figE.set_aspect('equal')
        figE.set_title('E direction')
        figphi.set_xlabel(r' $x/\delta x$')
        figE.set_xlabel(r' $x/\delta x$')
        figE.set_xlabel(r' $x/\delta x$')
        figphi.set_ylabel(r' $y/\delta x$')
        figE.set_ylabel(r' $y/\delta x$')
        figE.set_ylabel(r' $y/\delta x$')
        plt.tight_layout()

        #show and save
        fig.show()
        fig.savefig('phi_andE_jacobi_heatmaps')
        #plot & save the jacobi setup
        self.plotE_andPhi_v_d('jacobi50')



    def gauss_seidel_update(self, nruns = 1, omega = 1):
        """the swaggiest code everrrrr - checkerboards for gauss-seidel muahahahahaaaa
        omega = 1 corresponds to GS"""
       
        checkerboard = np.indices((self.size, self.size,self.size)).sum(axis=0) % 2
        nn = np.array([[[0,0,0] , [0,1,0], [0,0,0] ],\
                        [[0,1,0] , [1,0,1], [0,1,0] ],\
                        [[0,0,0] , [0,1,0], [0,0,0] ]] )

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


    def OR_explore_omegas(self,omegarange = [1.7, 1.99], nomegas = 50 ,runslim = 20000):
        omegas = np.linspace(omegarange[0],omegarange[1], nomegas )
        data = {'omega':[], 'nequil':[]}
        for omega in tqdm(omegas):
            self.clear_history()
            nequil = self.gauss_seidel_update(nruns = runslim, omega = omega)
            #OR_method_update(nruns = runslim, omega = omega, verbose=False)
            data['omega'].append(omega)
            data['nequil'].append(nequil)
        data = pd.DataFrame(data)
        data.to_csv('exploring_omega')
        fig, ax = plt.subplots()
        ax.scatter(data['omega'],data['nequil'], color = 'k')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$n_{eq}$')
        plt.show()
        fig.savefig('exploring_omega')

PS = Poisson(size =50)
#PS.jacobi_equilibration_andvisualize()
# PS.clear_history()
PS.GS_equilibration_andvisualize()
# PS.clear_history()
# PS.OR_equilibration_andvisualize(runslim = 20000)
# PS.clear_history() 
PS.OR_explore_omegas()           
