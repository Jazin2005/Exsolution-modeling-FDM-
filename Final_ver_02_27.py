import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

class Cahn_Hillard (object):

    def __init__ (self, nx, c_ab_0, T, itr, aniso, W_or_ab, W_ab_or, skip, x, y):
        # number of grids along x direction
        self.nx = nx 

        # number of grids along y direction
        self.ny = self.nx 
        self.dx, self.dy = 2.0e-9, 2.0e-9 # spacing of computational grids [m]
        self.c_ab_0 = c_ab_0 # average composition of Albite [atomic fraction]
        self.R = 8.314 # gas constant
        self.T = T # temperature [K]
        self.itr = itr# total number of iteration
        self.aniso = aniso # Anisotropy ratio

        self.W_or_ab = W_or_ab # [J/mol]
        self.W_ab_or = W_ab_or # [J/mol]
        self.ac = 3.0e-14 # gradient coefficient [Jm2/mol]
        self.Da = 1.0e-04*np.exp(-300000.0/self.R/self.T) # diffusion coefficient of Albite atom [m2/s]
        self.Db = 1.0e-04*np.exp(-300000.0/self.R/self.T) # diffusion coefficient of Orthoclase atom [m2/s]
        self.dt = (self.dx*self.dx/self.Da)*0.1
        self.skip = skip
        self.x = x
        self.y= y

    
    def __call__ (self):


        for i in range (0, self.itr+1):
            if i ==0:
                c = np.zeros((self.nx,self.ny))
                c_new = np.zeros((self.nx,self.ny))

                np.random.seed(0)
                c = self.c_ab_0 + np.random.rand(self.nx, self.ny)*0.01

                ANIM = animation.FFMpegFileWriter(fps=10, codec=None, extra_args=None, metadata=None)
                fig, ax = plt.subplots(figsize=(10,10))
                ax.invert_yaxis()
                ims = []
                im = plt.imshow(c, vmin=0, vmax=1, cmap="jet")
                co = plt.colorbar(shrink=0.85, pad = 0.02)
                ims.append([im])
                concentration =    c.reshape(c.shape[0]*c.shape[1],1)
                total_mu_chem_c =  c.reshape(c.shape[0]*c.shape[1],1)
                total_mu_grad_c =  c.reshape(c.shape[0]*c.shape[1],1)
                total_driving_force_new = c.reshape(c.shape[0]*c.shape[1],1)
                total_diff_pot = c.reshape(c.shape[0]*c.shape[1],1)

            else:
                c_new, mu_chem_c_new, mu_grad_c_new, driving_force_new,diff_pot_new = self.solver(c,c_new)
                c[:,:] = c_new[:,:]
                print(i)
                if i%self.skip==0:
                    concentration = np.hstack((concentration, c_new.reshape(c_new.shape[0]*c_new.shape[1],1)))
                    total_mu_chem_c = np.hstack((total_mu_chem_c, mu_chem_c_new.reshape(mu_chem_c_new.shape[0]*mu_chem_c_new.shape[1],1)))
                    total_mu_grad_c = np.hstack((total_mu_grad_c, mu_grad_c_new.reshape(mu_grad_c_new.shape[0]*mu_grad_c_new.shape[1],1)))
                    total_driving_force_new = np.hstack((total_driving_force_new, driving_force_new.reshape(driving_force_new.shape[0]*driving_force_new.shape[1],1)))
                    total_diff_pot = np.hstack((total_diff_pot, diff_pot_new.reshape(diff_pot_new.shape[0]*diff_pot_new.shape[1],1)))
                    im = plt.imshow(c, vmin=0, vmax=1, cmap="jet", animated = True)
                    ax.invert_yaxis()
                    ims.append([im])

        Movie = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        Movie.save("C:/Users/Javad/Desktop/result.mp4")
        self._plot_1D_concentration (concentration)
        self._plot_1D_driving_force (total_driving_force_new)
        self._plot_G(concentration)
        self.system_gibbs_energy (concentration)
        plt.show()
        return c

    def solver (self,c,c_new):
        mu_chem_c_new = np.zeros((self.nx,self.ny))
        mu_grad_c_new = np.zeros((self.nx,self.ny))
        driving_force_new = np.zeros((self.nx,self.ny))
        diff_pot_new = np.zeros((self.nx,self.ny))
        for j in range(self.ny):
            for i in range(self.nx):
                ip = i + 1
                im = i - 1
                jp = j + 1
                jm = j - 1
                ipp = i + 2
                imm = i - 2
                jpp = j + 2
                jmm = j - 2

                # Apply periodic boundary condition
                
                if ip > self.nx-1:  
                    ip = ip - self.nx
                if im < 0:
                    im = im + self.nx
                if jp > self.ny-1:
                    jp = jp - self.ny
                if jm < 0:
                    jm = jm + self.ny
                if ipp > self.nx-1: 
                    ipp = ipp - self.nx
                if imm < 0:
                    imm = imm + self.nx
                if jpp > self.ny-1:
                    jpp = jpp - self.ny
                if jmm < 0:
                    jmm = jmm + self.ny
                
                
                cc = c[i,j] # at (i,j) "centeral point"
                ce = c[ip,j] # at (i+1.j) "eastern point"
                cw = c[im,j] # at (i-1,j) "western point"
                cs = c[i,jm] # at (i,j-1) "southern point"
                cn = c[i,jp] # at (i,j+1) "northern point"
                cse = c[ip,jm] # at (i+1, j-1)
                cne = c[ip,jp]
                csw = c[im,jm]
                cnw = c[im,jp]
                cee = c[ipp,j] 
                cww = c[imm,j]
                css = c[i,jmm]
                cnn = c[i,jpp]
                
                # chemical term of chemical potential related to Gibbs energy (Analytical solution)
                #mu_chem_c = (np.log(cc)-np.log(1.0-cc)) + WRT*(1.0-2.0*cc) 
                #mu_chem_w = (np.log(cw)-np.log(1.0-cw)) + WRT*(1.0-2.0*cw) 
                #mu_chem_e = (np.log(ce)-np.log(1.0-ce)) + WRT*(1.0-2.0*ce) 
                #mu_chem_n = (np.log(cn)-np.log(1.0-cn)) + WRT*(1.0-2.0*cn)  
                #mu_chem_s = (np.log(cs)-np.log(1.0-cs)) + WRT*(1.0-2.0*cs)

                # chemical term of chemical potential related to Gibbs energy (Numerical solution)
                
                delt_c = 0.001
                mu_chem_c = (self.gibbs_func (cc + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (cc-delt_c, W_or_ab, W_ab_or))/(2*delt_c)
                mu_chem_w = (self.gibbs_func (cw + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (cw-delt_c, W_or_ab, W_ab_or))/(2*delt_c) 
                mu_chem_e = (self.gibbs_func (ce + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (ce-delt_c, W_or_ab, W_ab_or))/(2*delt_c)
                mu_chem_n = (self.gibbs_func (cn + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (cn-delt_c, W_or_ab, W_ab_or))/(2*delt_c)  
                mu_chem_s = (self.gibbs_func (cs + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (cs-delt_c, W_or_ab, W_ab_or))/(2*delt_c)
                

                mu_chem_c_A = (self.gibbs_func (1-cc + delt_c, W_or_ab, W_ab_or) - self.gibbs_func (1-cc-delt_c, W_or_ab, W_ab_or))/(2*delt_c)

                

                # gradient term of chemical potential related to concentration gradient
                mu_grad_c = -self.ac*( self.aniso*(ce -2.0*cc +cw )/self.dx/self.dx + (cn  -2.0*cc +cs )/self.dy/self.dy ) 
                mu_grad_w = -self.ac*( self.aniso*(cc -2.0*cw +cww)/self.dx/self.dx + (cnw -2.0*cw +csw)/self.dy/self.dy )
                mu_grad_e = -self.ac*( self.aniso*(cee-2.0*ce +cc )/self.dx/self.dx + (cne -2.0*ce +cse)/self.dy/self.dy )  
                mu_grad_n = -self.ac*( self.aniso*(cne-2.0*cn +cnw)/self.dx/self.dx + (cnn -2.0*cn +cc )/self.dy/self.dy ) 
                mu_grad_s = -self.ac*( self.aniso*(cse-2.0*cs +csw)/self.dx/self.dx + (cc  -2.0*cs +css)/self.dy/self.dy )              
                
                mu_grad_c_A = -self.ac*( self.aniso*((1-ce) -2.0*(1-cc) +(1-cw) )/self.dx/self.dx + ((1-cn)  -2.0*(1-cc) +(1-cs) )/self.dy/self.dy )

                driving_force = mu_chem_c + mu_grad_c -(mu_chem_c_A+mu_grad_c_A)
                # total chemical potential
                mu_c = mu_chem_c + mu_grad_c 
                mu_w = mu_chem_w + mu_grad_w 
                mu_e = mu_chem_e + mu_grad_e 
                mu_n = mu_chem_n + mu_grad_n 
                mu_s = mu_chem_s + mu_grad_s 
        
                nabla_mu = (mu_w -2.0*mu_c + mu_e)/self.dx/self.dx + (mu_n -2.0*mu_c + mu_s)/self.dy/self.dy    
                dc2dx2 = ((ce-cw)*(mu_e-mu_w))/(4.0*self.dx*self.dx)
                dc2dy2 = ((cn-cs)*(mu_n-mu_s))/(4.0*self.dy*self.dy) 
                
                DaDb = self.Da/self.Db

                # Mobility of A
                mob = (self.Da/self.R/self.T)*(cc+DaDb*(1.0-cc))*cc*(1.0-cc) 

                # gradient Mobility of A
                dmdc = (self.Da/self.R/self.T)*((1.0-DaDb)*cc*(1.0-cc)+(cc+DaDb*(1.0-cc))*(1.0-2.0*cc)) 
            
                dcdt = mob*nabla_mu + dmdc*(dc2dx2 + dc2dy2)
                c_new[i,j] = c[i,j] + dcdt *self.dt
                mu_chem_c_new[i,j] = mu_chem_c
                mu_grad_c_new[i,j] = mu_grad_c
                driving_force_new[i,j] = driving_force
                diff_pot_new[i,j] = mu_c
        
        return c_new, mu_chem_c_new, mu_grad_c_new, driving_force_new,diff_pot_new
    
    def gibbs_func (self, c, W_or_ab, W_ab_or):
        # orthoclase-albite
        c_ab = c
        c_or = 1.0-c
        gibbs_energy = self.R*self.T*(c_ab*np.log(c_ab) + c_or*np.log(c_or)) + W_or_ab*c_or*c_ab**2 + W_ab_or*c_ab*c_or**2

        return gibbs_energy

    def _plot_1D_concentration (self, c):

        
        c_1 = c[:,0]
        c_2 = c[:, 20]
        c_3 = c[:, 60]
        c_4 = c[:, 90]
        c_5 = c[:, 128]

        c_1_reshape = np.reshape(c_1, (self.nx, self.ny))
        c_2_reshape = np.reshape(c_2, (self.nx, self.ny))
        c_3_reshape = np.reshape(c_3, (self.nx, self.ny))
        c_4_reshape = np.reshape(c_4, (self.nx, self.ny))
        c_5_reshape = np.reshape(c_5, (self.nx, self.ny))
        
        fig, axes = plt.subplots(5,1,figsize=(20,50),constrained_layout = True)
        fig.suptitle('Concentration vs X Direction', fontsize=20)
        # Plot in x direction

        axes[0].plot(np.linspace(1, self.nx, self.ny),c_1_reshape[self.x,:], marker="o", color='r', label="Time step= {}".format(0))
        axes[1].plot(np.linspace(1, self.nx, self.ny),c_2_reshape[self.x,:], marker="o",  color='g', label="Time step= {}".format(2000))
        axes[2].plot(np.linspace(1, self.nx, self.ny),c_3_reshape[self.x,:], marker="o",  color='c', label="Time step= {}".format(6000))
        axes[3].plot(np.linspace(1, self.nx, self.ny),c_4_reshape[self.x,:], marker="o",  color='m', label="Time step= {}".format(9000))
        axes[4].plot(np.linspace(1, self.nx, self.ny),c_5_reshape[self.x,:], marker="o",  color='blue', label="Time step= {}".format(13000))
        axes[0].legend(prop={"size":13}, loc="lower right")
        axes[1].legend(prop={"size":13}, loc="lower right")
        axes[2].legend(prop={"size":13}, loc="lower right")
        axes[3].legend(prop={"size":13}, loc="lower right")
        axes[4].legend(prop={"size":13}, loc="lower right")

        axes[2].set_ylabel('Concentration', fontsize=15)
        axes[4].set_xlabel('X direction', fontsize=15)

        axes[0].tick_params(axis='both', labelsize=13)
        axes[1].tick_params(axis='both', labelsize=13)
        axes[2].tick_params(axis='both', labelsize=13)
        axes[3].tick_params(axis='both', labelsize=13)
        axes[4].tick_params(axis='both', labelsize=13)

        plt.setp(axes[0].get_xticklabels(), visible=False)
        plt.setp(axes[1].get_xticklabels(), visible=False)
        plt.setp(axes[2].get_xticklabels(), visible=False)
        plt.setp(axes[3].get_xticklabels(), visible=False)

        axes[0].set_ylim([0.1, 0.9])
        axes[1].set_ylim([0.1, 0.9])
        axes[2].set_ylim([0.1, 0.9])
        axes[3].set_ylim([0.1, 0.9])
        axes[4].set_ylim([0.1, 0.9])

    def _plot_1D_driving_force (self, total_driving_force_new):

        
        driving_force_1 = total_driving_force_new[:,0]
        driving_force_2 = total_driving_force_new[:, 3]
        driving_force_3 = total_driving_force_new[:, 8]
        driving_force_4 = total_driving_force_new[:, 50]
        driving_force_5 = total_driving_force_new[:, 128]

        driving_force_1_reshape = np.reshape(driving_force_1, (self.nx, self.ny))
        driving_force_2_reshape = np.reshape(driving_force_2, (self.nx, self.ny))
        driving_force_3_reshape = np.reshape(driving_force_3, (self.nx, self.ny))
        driving_force_4_reshape = np.reshape(driving_force_4, (self.nx, self.ny))
        driving_force_5_reshape = np.reshape(driving_force_5, (self.nx, self.ny))
        
        fig, axes = plt.subplots(5,1,figsize=(20,50),constrained_layout = True)
        fig.suptitle('Driving Force vs X Direction', fontsize=20)
        # Plot in x direction

        axes[0].plot(np.linspace(1, self.nx, self.ny),driving_force_1_reshape[self.x,:], marker="o", color='r', label="Time step= {}".format(0))
        axes[1].plot(np.linspace(1, self.nx, self.ny),driving_force_2_reshape[self.x,:], marker="o",  color='g', label="Time step= {}".format(300))
        axes[2].plot(np.linspace(1, self.nx, self.ny),driving_force_3_reshape[self.x,:], marker="o",  color='c', label="Time step= {}".format(800))
        axes[3].plot(np.linspace(1, self.nx, self.ny),driving_force_4_reshape[self.x,:], marker="o",  color='m', label="Time step= {}".format(5000))
        axes[4].plot(np.linspace(1, self.nx, self.ny),driving_force_5_reshape[self.x,:], marker="o",  color='blue', label="Time step= {}".format(13000))
        axes[0].legend(prop={"size":13}, loc="lower right")
        axes[1].legend(prop={"size":13}, loc="lower right")
        axes[2].legend(prop={"size":13}, loc="lower right")
        axes[3].legend(prop={"size":13}, loc="lower right")
        axes[4].legend(prop={"size":13}, loc="lower right")

        axes[2].set_ylabel('Albite Driving force ((mole*m)/sec⁻²)', fontsize=15)
        axes[4].set_xlabel('X direction', fontsize=15)

        axes[0].tick_params(axis='both', labelsize=13)
        axes[1].tick_params(axis='both', labelsize=13)
        axes[2].tick_params(axis='both', labelsize=13)
        axes[3].tick_params(axis='both', labelsize=13)
        axes[4].tick_params(axis='both', labelsize=13)

        plt.setp(axes[0].get_xticklabels(), visible=False)
        plt.setp(axes[1].get_xticklabels(), visible=False)
        plt.setp(axes[2].get_xticklabels(), visible=False)
        plt.setp(axes[3].get_xticklabels(), visible=False)

        axes[0].set_ylim([-2500,2500])
        axes[1].set_ylim([-2500,2500])
        axes[2].set_ylim([-2500,2500])
        axes[3].set_ylim([-2500,2500])
        axes[4].set_ylim([-2500,2500])

    def _plot_G(self, c):

        selected_c = c
        c1 = selected_c[self.x*self.y+1, :]

        fig, axes= plt.subplots()
        fig.suptitle('Spinodal Decompostion Process', fontsize=20)
        
        axes.set_ylabel('Gibbs Free energy (J)', fontsize=15)
        axes.set_xlabel('Concentration of Albite', fontsize=15)
        axes.tick_params(axis='both', labelsize=13)

        c1 = selected_c[self.x*self.y, :]
        c1_counter = 1-c1

        # Calculate gibbs energy
        gibbs_c1 = self.gibbs_func (c1, W_or_ab, W_ab_or)
        gibbs_c1_counter = self.gibbs_func (c1_counter, W_or_ab, W_ab_or)
  
        c = np.linspace(0.01, 0.99, 100)
        gibbs_en = self.gibbs_func (c, W_or_ab, W_ab_or)
        axes.plot(c, gibbs_en, color='g', linewidth=4, label="Gibss free energy")
        axes.plot(c1, gibbs_c1, color='b', marker="o", markersize=5, label="Gibbs energy minimization process of Albite")
        axes.quiver(c1[:-1], gibbs_c1[:-1], c1[1:]-c1[:-1], gibbs_c1[1:]-gibbs_c1[:-1], scale_units='xy', angles='xy', scale=0.5)
        axes.plot(c1_counter, gibbs_c1_counter, color='r', marker="o", markersize=5, label="Gibbs energy minimization process of Orthoclase")
        axes.quiver(c1_counter[:-1], gibbs_c1_counter[:-1], c1_counter[1:]-c1_counter[:-1], gibbs_c1_counter[1:]-gibbs_c1_counter[:-1], scale_units='xy', angles='xy', scale=0.5)
        axes.legend(prop={"size":13}, loc="upper center")
    
    def system_gibbs_energy (self, c):
        system_energy = self.gibbs_func (c, W_or_ab, W_ab_or)
        system_energy = np.sum(system_energy, axis=0)

        fig, ax= plt.subplots()
        fig.suptitle('System Free Energy', fontsize=20)
        ax.set_ylabel('System Free Energy (J)', fontsize=15)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        #system_free_energy = total_mu_chem_c[self.x*self.y,:]
        x = self.skip*np.linspace(1, system_energy.shape[0], num=system_energy.shape[0])
        print(system_energy.shape)
        ax.plot(x[0:-1], system_energy[0:-1], color='blue', marker="o", markersize=7)
        ax.legend(prop={"size":13}, loc="upper right")
    


# number of grids along x direction
nx = 100 

# number of grids along y direction
ny = nx 
dx, dy = 2.0e-9, 2.0e-9 # spacing of computational grids [m]
c_ab_0 = 0.55 # average composition of Albite [atomic fraction]
T = 500 # temperature [K]
P = 5 # Pressure [bar]
itr = 13000# total number of iteration
aniso = 1 # Anisotropy ratio

W_h_or_ab = 20083  # [J/mol]
W_h_ab_or = 20083  # [J/mol]

W_s_or_ab = 10.3
W_s_ab_or = 10.3

W_v_or_ab = 0.301
W_v_ab_or = 0.510

W_or_ab = W_h_or_ab - T*W_s_or_ab + P*W_v_or_ab
W_ab_or = W_h_ab_or - T*W_s_ab_or + P*W_v_ab_or

x = 75    # 23, 17 # The grid number in x direction for visualization. 
          # This variable should be less or equal than nx
y = 3    # The grid number in x direction for visualization.
          # This variable should be less or equal than ny

skip = 100
Eq = Cahn_Hillard(nx, c_ab_0, T, itr, aniso, W_or_ab, W_ab_or, skip, x, y)
simulation = Eq()