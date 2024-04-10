# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:19:06 2024

@author: samuel.delgado
"""
import lattpy as lp # https://lattpy.readthedocs.io/en/latest/tutorial/finite.html#position-and-neighbor-data
import matplotlib.pyplot as plt
from Site import Site,Island
from scipy import constants
import numpy as np
from matplotlib import cm


# Rotation of a vector - Is copper growing in [111] direction?
# The basis vector is in [001]
# https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python


class Crystal_Lattice():
    
    def __init__(self,lattice_properties,experimental_conditions,Act_E_list):
        self.lattice_constants = lattice_properties[0]
        self.crystal_size = lattice_properties[1]
        self.bravais_latt = lattice_properties[2]
        self.latt_orientation = lattice_properties[3]
        self.chemical_specie = experimental_conditions[4]
        self.temperature = experimental_conditions[3]
        self.mass_specie = experimental_conditions[2]
        self.activation_energies = Act_E_list
        self.time = 0
        self.list_time = []
        
        self.lattice_model()
        self.crystal_grid()
        self.calculate_crystallographic_planes()

        self.sites_occupied = [] # Sites occupy be a chemical specie
        self.adsorption_sites = [] # Sites availables for deposition or migration
        # Obtain all the positions in the grid that are supported by the
        # substrate or other deposited chemical species
        update_supp_av = [idx for idx in self.grid_crystal.keys()]
        update_specie_events = []
        self.update_sites(update_specie_events,update_supp_av)
        
        #Transition rate for adsortion of chemical species
        self.transition_rate_adsorption(experimental_conditions[0:4])
        
        
    # Model with all the possible lattice points
    def lattice_model(self):
        
        # Face centered cubic - 1 specie
        if self.bravais_latt == 'fcc':
            latt = lp.Lattice.fcc(self.lattice_constants[0])
            latt.add_atom()
            latt.add_connections(1)
            latt.analyze()
            latt.build((self.crystal_size[0], self.crystal_size[1],self.crystal_size[2]))
            self.latt = latt

            if self.latt_orientation == '001':
                self.basis_vectors = self.latt.vectors # Basis vectors
                self.positions_cartesian = self.latt.positions # Position in cartesian coordinates
                self.positions_idx = self.latt.indices[:,:3] # Position in lattices coordinates (indexes)
                
            elif self.latt_orientation == '111':
                basis_vectors = self.latt.vectors # Basis vectors

                # Rotate around z axis 45 degrees
                rotation_1 = ['z',np.pi/4]
                vectors_1 = [list(self.rotate_vector(vector,rotation_1[0],rotation_1[1])) for vector in basis_vectors]

                # We can calculate the angle establishing the condition that z components are the same for the three vectors 
                rotation_2 = ['x',np.arctan(vectors_1[2][2]/(vectors_1[0][1] - vectors_1[2][1]))]
                vectors_2 = [list(self.rotate_vector(vector,rotation_2[0],rotation_2[1])) for vector in vectors_1]
                self.basis_vectors = vectors_2

    # Initialize the crystal grid with no chemical species
    def crystal_grid(self):
        
        if self.latt_orientation == '001':
            # Dictionary with lattice coordinates (indexes) as keys and Site objects as values
            # Site includes:
            #  - Chemical specie - Empty
            #  - Position in cartesian coordinates
            self.grid_crystal = {
                tuple(idx):Site("Empty",
                    tuple(pos),
                    self.activation_energies)
                for idx, pos in zip(self.positions_idx, self.positions_cartesian)
            }
            
            # Pruning the neighbors according to the lattice domain
            # E.g.: (0,0,0) only has 3 neighbors within the domain
            #  - Nearest neighbors indexes
            #  - Nearest neighbors cartesian
            for idx,site in self.grid_crystal.items():
                site.neighbors_analysis(self.grid_crystal,self.latt.get_neighbors(idx)[:,:3],self.latt.get_neighbor_positions(idx))
        
        elif self.latt_orientation == '111':

            start_idx_site = (0,0,0) # Starting site
            self.grid_crystal = {start_idx_site:Site("Empty",
                (0,0,0),
                self.activation_energies)}
            
            # We employ a Depth-First Search algorithm to build the grid_crystal
            # based on the nearest neighbors of each site
            # -----------------------------------------------------------------------
            # Recursive is only valid for small crystal structures --> recursion depth error
            # visited = set()
            # self.dfs_recursive(start_idx_site, visited))
            # -----------------------------------------------------------------------
            self.dfs_iterative(start_idx_site)
            
            self.positions_idx = list(self.grid_crystal.keys())
            self.positions_cartesian = [site.position for site in self.grid_crystal.values()]
                
            for idx,site in self.grid_crystal.items():
                # Neighbors for each idx in grid_crystal
                neighbors_positions = [self.idx_to_cart(idx) for idx in self.latt.get_neighbors(idx)[:,:3]]
                site.neighbors_analysis(self.grid_crystal,self.latt.get_neighbors(idx)[:,:3],neighbors_positions,self.crystal_size)
        
    def calculate_crystallographic_planes(self):
        
# =============================================================================
#         Han, Yong, Feng Liu, Shao-Chun Li, Jin-Feng Jia, Qi-Kun Xue, and Byeong-Joo Lee. 
#         "Kinetics of mesa overlayer growth: Climbing of adatoms onto the mesa top." 
#         Applied Physics Letters 92, no. 2 (2008).
# 
#         Good figure for the crystallographic planes
# 
#         Good link:
#         https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Surface_Science_(Nix)/01%3A_Structure_of_Solid_Surfaces/1.03%3A_Surface_Structures-_fcc_Metals
# =============================================================================
        # Crystallographic planes with (111) orentation
        n_plane = np.cross(self.basis_vectors[0],self.basis_vectors[1])
        crystallographic_planes = [((111),n_plane/np.linalg.norm(n_plane))]
        
        # Crystallographic planes with (111) orentation
        n_plane = np.cross(self.basis_vectors[0],self.basis_vectors[2])
        crystallographic_planes.append(((111),n_plane/np.linalg.norm(n_plane)))
        
        # Crystallographic planes with (111) orentation
        n_plane = np.cross(self.basis_vectors[1],self.basis_vectors[2])
        crystallographic_planes.append(((111),n_plane/np.linalg.norm(n_plane)))
        
        # Crystallographic planes with (111) orentation
        crystallographic_planes.append(((111),np.array([0,0,1])))
                
        # Select the first element far from the boundaries
        for idx,site in self.grid_crystal.items():
            if ((self.lattice_constants[0] < site.position[0] < self.crystal_size[0] - self.lattice_constants[0]) and
                (self.lattice_constants[1] < site.position[1] < self.crystal_size[1] - self.lattice_constants[1])):
                    site_idx = idx
                    break
        
        # Crystallographic planes with (100) orentation - (001) is equivalent
        for v in self.basis_vectors:
        
            for neighbor in self.grid_crystal[site_idx].migration_paths['Plane']:
                v_aux = np.array(self.idx_to_cart(neighbor[0])) - np.array(self.grid_crystal[site_idx].position)
                if round(self.angle_between(v,v_aux) - np.pi /2,2) == 0.0:
                    v2 = np.array(self.idx_to_cart(neighbor[0])) - np.array(self.grid_crystal[site_idx].position)
                    
            n_plane = np.cross(v,v2)
            crystallographic_planes.append(((100),n_plane/np.linalg.norm(n_plane)))
            
        self.crystallographic_planes = crystallographic_planes
            
        
    def available_adsorption_sites(self, update_supp_av = []):
        
        
        if update_supp_av == []:
            for idx,site in self.grid_crystal.items():
                if ('Substrate' in site.supp_by or len(site.supp_by) > 2) and (site.chemical_specie == 'Empty'):
                    self.adsorption_sites.append(idx)
        else:
            for idx in update_supp_av:
                if (idx in self.adsorption_sites) and (('Substrate' not in self.grid_crystal[idx].supp_by and len(self.grid_crystal[idx].supp_by) < 3) or (self.grid_crystal[idx].chemical_specie != 'Empty')):
                    self.adsorption_sites.remove(idx)
                    
                elif (idx not in self.adsorption_sites) and ('Substrate' in self.grid_crystal[idx].supp_by or len(self.grid_crystal[idx].supp_by) > 2) and self.grid_crystal[idx].chemical_specie == 'Empty':
                    self.adsorption_sites.append(idx)
                    
        
                    
    def transition_rate_adsorption(self,experimental_conditions):
# =============================================================================
#         Kim, S., An, H., Oh, S., Jung, J., Kim, B., Nam, S. K., & Han, S. (2022).
#         Atomistic kinetic Monte Carlo simulation on atomic layer deposition of TiN thin film. 
#         Computational Materials Science, 213. https://doi.org/10.1016/j.commatsci.2022.111620
# =============================================================================
        
        # Maxwell-Boltzman statistics for transition rate of adsorption rate
        sticking_coeff, partial_pressure, mass_specie, T = experimental_conditions
        
        # The mass in kg of a unit of the chemical specie
        mass_specie = mass_specie / constants.Avogadro / 1000
        
        # If the substrate surface is pristine, we take the supported sites,
        # which are actually those ones supported by the substrate
        if len(self.sites_occupied) == 0:
            n_sites_layer_0 = len(self.adsorption_sites)
        else:
            # Otherwise, we count the sites in the layer 0
            n_sites_layer_0 = 0
            for site in self.grid_crystal.values():
                if site.position[2] == 0:
                    n_sites_layer_0 += 1


        # Area per site = total area of the crystal / number of sites available at the bottom layer
        # Area in m^2
        area_specie = 1e-18 *self.crystal_size[0] * self.crystal_size[1] / n_sites_layer_0
        
        # Boltzmann constant (m^2 kg s^-2 K^-1)
        self.TR_ad = sticking_coeff * partial_pressure * area_specie / np.sqrt(2 * constants.pi * mass_specie * constants.Boltzmann * T)
    
    def limit_kmc_timestep(self,P_limits):
        
        self.timestep_limits = -np.log(1-P_limits)/self.TR_ad
        
    
# =============================================================================
#             Introduce particle
# =============================================================================
    def introduce_specie_site(self,idx,update_specie_events,update_supp_av):
        
        # Chemical specie deposited
        self.grid_crystal[idx].introduce_specie(self.chemical_specie)
        # Track sites occupied
        self.sites_occupied.append(idx) 
        # Track sites available
        update_specie_events.append(idx)
        
        # CAREFUL! We don't update 2nd nearest neighbors
        # Nodes we need to update
        update_supp_av.update(self.grid_crystal[idx].nearest_neighbors_idx)
        update_supp_av.add(idx) # Update the new specie to calculate its supp_by

        return update_specie_events,update_supp_av
    
# =============================================================================
#             Remove particle 
# =============================================================================
    def remove_specie_site(self,idx,update_specie_events,update_supp_av):
        
        # Chemical specie removed
        self.grid_crystal[idx].remove_specie()
        # Track sites occupied
        self.sites_occupied.remove(idx) 
        # Track sites available
        update_specie_events.remove(idx)
        
        # CAREFUL! We don't update 2nd nearest neighbors
        # Nodes we need to update
        update_supp_av.update(self.grid_crystal[idx].nearest_neighbors_idx)
        update_supp_av.add(idx) # Update the new empty space to calculate its supp_by
        
        return update_specie_events,update_supp_av

    def deposition_specie(self,t,rng,test = 0):
        

        update_supp_av = set()
        # We need to do a copy if we don't want to modify self.sites_occupied when
        # we modify update_specie_events
        # update_specie_events = (self.sites_occupied).copy()
        update_specie_events = []
        
        if test == 0:
            
            P = 1-np.exp(-self.TR_ad*t) # Adsorption probability in time t
            # Indexes of sites availables: supported by substrates or other species
            for idx in self.adsorption_sites:
                if rng.random() < P:   
                    # Introduce specie in the site
                    update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)

            

        # Single particle in a determined place
        elif test == 1:
            
            if self.latt_orientation == '001': idx = (3,2,-2)
            elif self.latt_orientation == '111': idx = (1,11,-12)
                
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
                
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
        # Single particle introduced and removed
        elif test == 2:
            
            if self.latt_orientation == '001': idx = (3,2,-2)
            elif self.latt_orientation == '111': idx = (1,11,-12)
            
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
                 
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
            print('Sites occupied: ', self.sites_occupied)
            print('Number of sites availables: ', len(self.adsorption_sites))
            print('Possible events: ', self.grid_crystal[idx].site_events)
            # Remove particle
            update_specie_events,update_supp_av = self.remove_specie_site(idx,update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
                 
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
            print('Sites occupied: ', self.sites_occupied)
            print('Number of sites availables: ', len(self.adsorption_sites))
            print('Possible events: ', self.grid_crystal[idx].site_events)

        # Introduce two adjacent particles
        elif test == 3:
            
            if self.latt_orientation == '001': idx = (3,2,-2)
            elif self.latt_orientation == '111': idx = (1,11,-12)
            
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
            
            neighbor = self.grid_crystal[idx].migration_paths['Plane'][0]
            # Introduce specie in the neighbor site
            update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
            

        # Hexagonal seed - 7 particles in plane
        elif test == 4:
            
            update_supp_av = set()
            update_specie_events = []
            idx = self.adsorption_sites[46]
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)

            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
            idx_neighbor_plane = self.grid_crystal[neighbor[0]].migration_paths['Plane'][1][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_plane,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
        
        # Hexagonal seed - 7 particles in plane, one on top
        elif test == 5:
            
            update_supp_av = set()
            update_specie_events = []
            idx = self.adsorption_sites[46]
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)

            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
            idx_neighbor_top = self.grid_crystal[neighbor[0]].migration_paths['Up'][0][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_top,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
            
        # 2 hexagonal seeds - 2 layers and one particle on the top 
        elif test == 6:

            update_supp_av = set()
            update_specie_events = []
            idx = self.adsorption_sites[46]
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)

            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
            idx_neighbor_top = self.grid_crystal[idx].migration_paths['Up'][0][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_top,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
            for neighbor in self.grid_crystal[idx_neighbor_top].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
            
            idx_neighbor_top = self.grid_crystal[neighbor[0]].migration_paths['Up'][0][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_top,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
            
        elif test == 7:
            
            update_supp_av = set()
            update_specie_events = []
            idx = self.adsorption_sites[46]
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)

            # Cluster in contact with the substrate
            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
            # Particle next to the cluster
            idx_neighbor_plane = self.grid_crystal[neighbor[0]].migration_paths['Plane'][1][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_plane,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
                
            # Cluster over the copper
            for neighbor in self.grid_crystal[idx].migration_paths['Up']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
        elif test == 8:
            from collections import deque

            # Create a deque object for the queue
            queue = deque()
            queue.append(self.adsorption_sites[52])
            visited = set()
            cluster_size = 29
            
            self.bfs_cluster(queue,visited,cluster_size)
            

            
    def processes(self,chosen_event):
 
        site_affected = self.grid_crystal[chosen_event[-1]]
        update_supp_av = set()
        update_specie_events = [chosen_event[-1]]
        

# =============================================================================
#         Specie migration
# =============================================================================
        if chosen_event[2] < site_affected.num_mig_path:
            
            # print('Support balance: ',len(self.grid_crystal[chosen_event[-1]].supp_by)-len(self.grid_crystal[chosen_event[1]].supp_by))
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(chosen_event[1],update_specie_events,update_supp_av)
            
            # Remove particle
            update_specie_events,update_supp_av = self.remove_specie_site(chosen_event[-1],update_specie_events,update_supp_av)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,update_supp_av)
  
    
    def update_sites(self,update_specie_events,update_supp_av):

        if update_supp_av:
            # There are new sites supported by the deposited chemical species
            # For loop over neighbors
            for idx in update_supp_av:
                self.grid_crystal[idx].supported_by(self.grid_crystal,self.crystallographic_planes)
                self.available_adsorption_sites(update_supp_av)
                if self.grid_crystal[idx].chemical_specie != 'Empty':
                    update_specie_events.append(idx)
        
        
        if update_specie_events: 
            # Sites are not available because a particle has migrated there
            for idx in update_specie_events:
                self.grid_crystal[idx].available_migrations(self.grid_crystal,idx)
                self.grid_crystal[idx].transition_rates(self.temperature)

    def track_time(self,t):
        
        self.time += t
        
    def add_time(self):
        
        self.list_time.append(self.time)

# =============================================================================
# --------------------------- PLOTTING FUNCTIONS ------------------------------
#         
# =============================================================================


    def plot_lattice_points(self,azim = 60,elev = 45):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x,y,z = zip(*self.positions_cartesian)
        
        ax.scatter3D(x, y, z, c='blue', marker='o')
        ax.set_aspect('equal', 'box')
        ax.view_init(azim=azim, elev = elev)

        ax.set_xlabel('x-axis (nm)')
        ax.set_ylabel('y-axis (nm)')
        ax.set_zlabel('z-axis (nm)')
        
        plt.show()
        
        
    def plot_crystal(self,azim = 60,elev = 45,path = '',i = 0):
        
        nr = 1
        nc = 2
        fig = plt.figure(constrained_layout=True,figsize=(15, 8),dpi=300)
        subfigs = fig.subfigures(nr, nc, wspace=0.1, hspace=7, width_ratios=[1,1])
        
        axa = subfigs[0].add_subplot(111, projection='3d')
        positions = np.array([self.grid_crystal[idx].position for idx in self.sites_occupied])
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        axa.scatter3D(x, y, z, c='blue', marker='o')
        
        axa.set_xlabel('x-axis (nm)')
        axa.set_ylabel('y-axis (nm)')
        axa.set_zlabel('z-axis (nm)')
        axa.view_init(azim=azim, elev = elev)

        axa.set_xlim([0, self.crystal_size[0]]) 
        axa.set_ylim([0, self.crystal_size[1]])
        axa.set_zlim([0, 2*self.crystal_size[2]])
        axa.set_aspect('equal', 'box')
        
        axb = subfigs[1].add_subplot(111, projection='3d')
        axb.scatter3D(x, y, z, c='blue', marker='o')
        
        axb.set_xlabel('x-axis (nm)')
        axb.set_ylabel('y-axis (nm)')
        axb.set_zlabel('z-axis (nm)')
        axb.view_init(azim=45, elev = 10)

        axb.set_xlim([0, self.crystal_size[0]]) 
        axb.set_ylim([0, self.crystal_size[1]])
        axb.set_zlim([0, 2*self.crystal_size[2]])
        axb.set_aspect('equal', 'box')


        if path == '':
            plt.show()
        else:
            plt.savefig(path+str(i)+'_t(s) = '+str(round(self.time,5))+' .png', dpi = 300)
            plt.clf()
        plt.show()
        
    def plot_crystal_surface(self):
        
        x,y,z = self.obtain_surface_coord()
                
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
        
        # Set labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_zlim([0, self.crystal_size[2]])
        
        ax.view_init(azim=45, elev = 45)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Show the plot
        plt.show()
        
# =============================================================================
# --------------------------- MEASUREMENTS ------------------------------------
#         
# =============================================================================

    def measurements_crystal(self):
        
        self.calculate_mass()
        self.sites_occupation()
        self.average_thickness()
        self.terrace_area()
        self.RMS_roughness()
        
    def calculate_mass(self):
        
        x_size, y_size = self.crystal_size[:2]
        density = len(self.sites_occupied) * self.mass_specie / (x_size * y_size)
        g_to_ng = 1e9
        nm_to_cm = 1e7
        
        self.mass_gained = nm_to_cm**2 * g_to_ng * density / constants.Avogadro # (ng/cm2)

# =============================================================================
# We calculate % occupy per layer
# Average the contribution of each layer to the thickness acording to the z step
# =============================================================================
    def average_thickness(self):
        
        grid_crystal = self.grid_crystal
        z_step = self.basis_vectors[0][2]
        z_steps = int(self.crystal_size[2]/z_step + 1)
        layers = [0] * z_steps  # Initialize each layer separately
        
        for site in grid_crystal.values():
            z_idx = int(round(site.position[2] / z_step))
            layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0
            
        sites_per_layer = len(grid_crystal)/z_steps
        normalized_layers = [count / sites_per_layer for count in layers]
        # Number of sites occupied and percentage of occupation for each layer
        self.layers = [layers, normalized_layers]
        
        # Layer 0 is z = 0, so it doesn't contribute
        self.thickness = sum(normalized_layers[1:]) * z_step # (nm)    
        
    def sites_occupation(self):
        
        self.fraction_sites_occupied = len(self.sites_occupied) / len(self.grid_crystal) 
        
    def terrace_area(self):
        
        layers = self.layers[0]
        grid_crystal = self.grid_crystal
        z_step = self.basis_vectors[0][2]
        z_steps = int(self.crystal_size[2]/z_step + 1)
        sites_per_layer = len(grid_crystal)/z_steps

        area_per_site = self.crystal_size[0] * self.crystal_size[1] / sites_per_layer
        
        terraces = [(layers[i-1] - layers[i]) * area_per_site for i in range(1,len(layers))]
        terraces.append(layers[-1] * area_per_site) # (nm2)
        
        self.terraces = terraces
        
    def RMS_roughness(self):
        
        x,y,z = self.obtain_surface_coord()
        z_mean = np.mean(z)
        self.surf_roughness_RMS = np.sqrt(np.mean((np.array(z)-z_mean)**2))
    
    def islands_analysis(self):

        visited = set()
        normalized_layers = self.layers[1]
        count_islands = [0] * len(normalized_layers)
        layers_no_complete = np.where(np.array(normalized_layers) != 1.0)
        count_islands[normalized_layers == 1] = 1
        z_step = self.basis_vectors[0][2]

        islands_list = []

        for z_idx in layers_no_complete[0]:    
            z_layer = round(z_idx * z_step,3)
            for idx_site in self.sites_occupied:     
                if self.grid_crystal[idx_site].position[2] == z_layer: 
                    island_slice = set()
                    visited,island_slice = self.detect_islands(idx_site,visited,island_slice,self.chemical_specie)

                    if len(island_slice):
                        island_sites = island_slice.copy()
                        visited,island_sites = self.build_island(visited,island_sites,list(island_slice)[0],self.chemical_specie)
                        islands_list.append(Island(z_idx,z_layer,island_sites))
                        count_islands[z_idx] += 1
                        
        self.islands_list = islands_list

        
# =============================================================================
#     Auxiliary functions
#     
# =============================================================================
    def unit_vector(self,vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """Finds angle between two vectors"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # Function to rotate a vector
    def rotate_vector(self,vector, axis, theta):
        """
        Rotates a 3D vector around a specified axis.
    
        Parameters:
        - vector: The 3D vector to rotate.
        - axis: The axis of rotation ('x', 'y', or 'z').
        - theta: The rotation angle in radians.
    
        Returns:
        The rotated vector.
        """
        if axis == 'x':
            R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 'z':
            R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        else:
            raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")
        
        return np.dot(R, vector)
    
    # Depth-First Search - Traverse a network or a graph -> grid_crystal
    def dfs_recursive(self, idx_site, visited):
        # We calculate the cartesian coordinates of the site using the basis vectors
        cart_site = self.idx_to_cart(idx_site)
        # cart_site[2] >= -1e-3 to avoid that some sites in the zero layer get outside
        if idx_site not in visited and (cart_site[0] >= 0 and cart_site[0] <= self.crystal_size[0]) and (cart_site[1] >= 0 and cart_site[1] <= self.crystal_size[1]) and (cart_site[2] >= -1e-3 and cart_site[2] <= self.crystal_size[2]):
            # We track the created sites
            visited.add(idx_site)
            # We create the site with the cartesian coordinates
            self.grid_crystal[idx_site] = Site("Empty",
                tuple(cart_site),
                self.activation_energies)
            
            for neighbor in self.latt.get_neighbors(idx_site):
                self.dfs_recursive(tuple(neighbor[:3]), visited)
                
    def dfs_iterative(self, start_idx_site):
        visited = set()
        stack = [start_idx_site]
    
        while stack:
            current_idx_site = stack.pop()
            if current_idx_site in visited:
                continue
    
            # Calculate the cartesian coordinates of the site using the basis vectors
            cart_site = self.idx_to_cart(current_idx_site)
   
            
            if (
                0 <= cart_site[0] <= self.crystal_size[0]
                and 0 <= cart_site[1] <= self.crystal_size[1]
                and 0 <= cart_site[2] <= self.crystal_size[2]
            ):
                # Track the created site
                visited.add(current_idx_site)
                # Create the site with the cartesian coordinates
                self.grid_crystal[current_idx_site] = Site(
                    "Empty", tuple(cart_site), self.activation_energies
                )
    
                # Push neighbors onto the stack
                stack.extend(tuple(neighbor[:3]) for neighbor in self.latt.get_neighbors(current_idx_site))
    # Breadth-First Search (Recursive) - Traverse a network or a graph -> grid_crystal
    # to build a cluster of a certain size
    def bfs_cluster(self,queue,visited,cluster_size):
        
        if not queue or len(visited) >= cluster_size:
            return
        
        # Dequeue a site from the front of the queue
        #Starting point
        current_idx_site = queue.popleft()
        
        if current_idx_site not in visited:
            visited.add(current_idx_site)
            update_supp_av = set()
            update_specie_events = []
            
            update_specie_events,update_supp_av = self.introduce_specie_site(current_idx_site,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
            
            # Enqueue all unvisited neighbors of the current site
            for neighbor in self.grid_crystal[current_idx_site].migration_paths['Plane']:
                if neighbor[0] not in visited:
                    queue.append(neighbor[0])
                
        # Recur to process the next site in the queue
        self.bfs_cluster(queue, visited, cluster_size)
        
    def idx_to_cart(self,idx):
        return tuple(round(element,3) for element in np.sum(idx * np.transpose(self.basis_vectors), axis=1))
    
    
# =============================================================================
#     Function to detect island and the coordinates of the base
# =============================================================================
    def detect_islands(self,idx_site,visited,island_slice,chemical_specie):

        site = self.grid_crystal[idx_site] 
        
        if idx_site not in visited and site.chemical_specie == chemical_specie:
            visited.add(idx_site)
            island_slice.add(idx_site)
            # dfs_recursive
            for idx in site.migration_paths['Plane']:
                visited,island_slice = self.detect_islands(idx[0],visited,island_slice,chemical_specie)
                                       
        return visited,island_slice

# =============================================================================
#     Function to build the full island starting from the base obtained in detect_islands()
# =============================================================================
    def build_island(self,visited,island_sites,idx,chemical_specie):
          
        site = self.grid_crystal[idx]
            
        for element in site.migration_paths['Up'] + site.migration_paths['Plane']:
    
            if element[0] not in visited and self.grid_crystal[element[0]].chemical_specie == chemical_specie:
                visited.add(element[0])
                island_sites.add(element[0])
                visited,island_sites = self.build_island(visited,island_sites,element[0],chemical_specie)
                
        return visited,island_sites
    
    def obtain_surface_coord(self):
        
        grid_crystal = self.grid_crystal
        
        x = []
        y = []
        z = []
        
        for site in grid_crystal.values():
            top_layer_empty_sites = 0
            for jump in site.migration_paths['Up']:
                if grid_crystal[jump[0]].chemical_specie == 'Empty': top_layer_empty_sites +=1
                     
            if (site.chemical_specie != 'Empty') and top_layer_empty_sites >= 2:
                x.append(site.position[0])
                y.append(site.position[1])
                z.append(site.position[2])
                
        return x,y,z