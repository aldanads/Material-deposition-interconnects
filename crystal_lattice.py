# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:19:06 2024

@author: samuel.delgado
"""
import lattpy as lp # https://lattpy.readthedocs.io/en/latest/tutorial/finite.html#position-and-neighbor-data
import matplotlib.pyplot as plt
from Site import Site
from scipy import constants
import numpy as np

# Rotation of a vector - Is copper growing in [111] direction?
# The basis vector is in [001]
# https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python


class Crystal_Lattice():
    
    def __init__(self,lattice_properties,experimental_conditions,E_mig):
        self.lattice_constants = lattice_properties[0]
        self.crystal_size = lattice_properties[1]
        self.bravais_latt = lattice_properties[2]
        self.latt_orientation = lattice_properties[3]
        self.chemical_specie = experimental_conditions[4]
        self.activation_energies = E_mig
        self.time = 0
        self.list_time = []
        
        self.lattice_model()
        self.crystal_grid()
        
        self.sites_occupied = [] # Sites occupy be a chemical specie
        self.adsorption_sites = [] # Sites availables for deposition or migration
        # Obtain all the positions in the grid that are supported by the
        # substrate or other deposited chemical species
        self.available_adsorption_sites() 
        
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
            """
            Need to debug
            """
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
                self.grid_crystal[idx].supported_by(self.grid_crystal)
                self.available_adsorption_sites(update_supp_av)
                if self.grid_crystal[idx].chemical_specie != 'Empty':
                    update_specie_events.append(idx)
        
        
        if update_specie_events: 
            # Sites are not available because a particle has migrated there
            for idx in update_specie_events:
                self.grid_crystal[idx].available_migrations(self.grid_crystal)
                self.grid_crystal[idx].transition_rates()

    def track_time(self,t):
        
        self.time += t
        
    def add_time(self):
        
        self.list_time.append(self.time)



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
        
        
    def plot_crystal(self,azim = 60,elev = 45):
        
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111, projection='3d')
           
        positions = np.array([self.grid_crystal[idx].position for idx in self.sites_occupied])
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.scatter3D(x, y, z, c='blue', marker='o')
        
        ax.set_xlabel('x-axis (nm)')
        ax.set_ylabel('y-axis (nm)')
        ax.set_zlabel('z-axis (nm)')
        ax.view_init(azim=azim, elev = elev)

        ax.set_xlim([0, self.crystal_size[0]]) 
        ax.set_ylim([0, self.crystal_size[1]])
        ax.set_zlim([0, 2*self.crystal_size[2]])
        ax.set_aspect('equal', 'box')

        plt.show()
        
# =============================================================================
#     Auxiliary functions
#     
# =============================================================================
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
    
    def idx_to_cart(self,idx):
        return tuple(round(element,3) for element in np.sum(idx * np.transpose(self.basis_vectors), axis=1))