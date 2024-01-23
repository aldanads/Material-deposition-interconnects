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


class Crystal_Lattice():
    
    def __init__(self,lattice_constants,crystal_size,bravais_latt,experimental_conditions,E_mig):
        
        self.lattice_constants = lattice_constants
        self.bravais_latt = bravais_latt
        self.crystal_size = crystal_size
        self.chemical_specie = experimental_conditions[4]
        self.activation_energies = E_mig
        self.time = 0

        
        self.latt = self.lattice_model()
        self.basis_vectors = self.latt.vectors # Basis vectors
        self.positions_cartesian = self.latt.positions # Position in cartesian coordinates
        self.positions_idx = self.latt.indices[:,:3] # Position in lattices coordinates (indexes)
        
        self.crystal_grid()
        
        self.sites_occupied = [] # Sites occupy be a chemical specie
        self.sites_available = [] # Sites availables for deposition or migration
        # Obtain all the positions in the grid that are supported by the
        # substrate or other deposited chemical species
        self.available_sites() 
        
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
            return latt
        
    # Initialize the crystal grid with no chemical species
    def crystal_grid(self):
        
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
        for idx,node in self.grid_crystal.items():
            node.neighbors_analysis(self.grid_crystal,self.latt.get_neighbors(idx)[:,:3],self.latt.get_neighbor_positions(idx))
        
    def available_sites(self):
        
        for idx,node in self.grid_crystal.items():
            if (len(node.supp_by) > 0) and (node.chemical_specie == 'Empty'):
                self.sites_available.append(idx)
                
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
            n_sites_layer_0 = len(self.sites_available)
        else:
            # Otherwise, we count the sites in the layer 0
            n_sites_layer_0 = 0
            for node in self.grid_crystal.values():
                if node.position[2] < self.lattice_constants[2] * 0.1:
                    n_sites_layer_0 += 1


        # Area per site = total area of the crystal / number of sites available at the bottom layer
        # Area in m^2
        area_specie = 1e-18 *self.crystal_size[0] * self.crystal_size[1] / n_sites_layer_0
        
        # Boltzmann constant (m^2 kg s^-2 K^-1)
        self.TR_ad = sticking_coeff * partial_pressure * area_specie / np.sqrt(2 * constants.pi * mass_specie * constants.Boltzmann * T)
    
    
# =============================================================================
#             Introduce particle
# =============================================================================
    def introduce_specie_site(self,idx,sites_not_available,need_to_update):
        
        # Chemical specie deposited
        self.grid_crystal[idx].introduce_specie(self.chemical_specie)
        self.sites_available.remove(idx)
        # Track sites occupied
        self.sites_occupied.append(idx) 
        # Track sites available
        sites_not_available.append(idx)
        # Nodes we need to update
        need_to_update.update(self.grid_crystal[idx].nearest_neighbors_idx)
        
        return sites_not_available,need_to_update
    
# =============================================================================
#             Remove particle 
# =============================================================================
    def remove_specie_site(self,idx,sites_not_available,need_to_update):
        # Chemical specie removed
        self.grid_crystal[idx].remove_specie()
        # Track sites occupied
        self.sites_occupied.remove(idx) 
        # Track sites available
        sites_not_available.remove(idx)
        self.sites_available.append(idx)
        # Nodes we need to update
        need_to_update.update(self.grid_crystal[idx].nearest_neighbors_idx)
        
        return sites_not_available,need_to_update

    def deposition_specie(self,t,rng,test = 0):
        
        need_to_update = set()
        # We need to do a copy if we don't want to modify self.sites_occupied when
        # we modify sites_not_available
        sites_not_available = (self.sites_occupied).copy()
        
        if test == 0:
            
            # Indexes of sites availables: supported by substrates or other species
            for idx in self.sites_available:
                P = 1-np.exp(-self.TR_ad*t) # Adsorption probability in time t
                if rng.random() < P:   
                    # Introduce specie in the site
                    sites_not_available,need_to_update = self.introduce_specie_site(idx,sites_not_available,need_to_update)

            # Update sites availables, the support to each site and available migrations
            self.update_sites(sites_not_available,need_to_update)
            

        # Single particle in a determined place
        elif test == 1:
            
            idx = (3,2,-2)
            
            # Introduce specie in the site
            sites_not_available,need_to_update = self.introduce_specie_site(idx,sites_not_available,need_to_update)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(sites_not_available,need_to_update)
                
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
        # Single particle that disappear or migrate
        elif test == 2:

            idx = (3,2,-2)
            
            # Introduce specie in the site
            sites_not_available,need_to_update = self.introduce_specie_site(idx,sites_not_available,need_to_update)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(sites_not_available,need_to_update)
                 
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
            print('Sites occupied: ', self.sites_occupied)
            print('Number of sites availables: ', len(self.sites_available))
            print('Possible events: ', self.grid_crystal[idx].site_events)

            # Remove particle
            sites_not_available,need_to_update = self.remove_specie_site(idx,sites_not_available,need_to_update)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(sites_not_available,need_to_update)
                 
            print('Particle in position: ',idx, ' is a ', self.grid_crystal[idx].chemical_specie)
            print('Neighbors of that particle: ', self.grid_crystal[idx].nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in self.grid_crystal[idx].nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
            print('Sites occupied: ', self.sites_occupied)
            print('Number of sites availables: ', len(self.sites_available))
            print('Possible events: ', self.grid_crystal[idx].site_events)


    
    def processes(self,chosen_event):
        
        site_affected = self.grid_crystal[chosen_event[-1]]
        need_to_update = set()
        sites_not_available = [chosen_event[-1]]
        

# =============================================================================
#         Specie migration
# =============================================================================
        if chosen_event[2] < site_affected.num_mig_path:
            
            # Introduce specie in the site
            sites_not_available,need_to_update = self.introduce_specie_site(chosen_event[1],sites_not_available,need_to_update)
            
            # Remove particle
            sites_not_available,need_to_update = self.remove_specie_site(chosen_event[-1],sites_not_available,need_to_update)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(sites_not_available,need_to_update)
  
    
    def update_sites(self,sites_not_available,need_to_update):
        
        
        if len(need_to_update) > 0:
            # There are new sites supported by the deposited chemical species
            # For loop over neighbors
            for idx in need_to_update:
                self.grid_crystal[idx].supported_by(self.grid_crystal)
                
                # If the site is supported by at least one specie, it becames 
                # available
                if len(self.grid_crystal[idx].supp_by) > 0:
                    self.sites_available.append(idx)
                
        if len(sites_not_available) > 0: 
            # Function filter() use a function and an iterable. 
            # When the function is true, take the element from iterable
            # Remove elements from "sites_not_available" that are in "sites_available"
            #self.sites_available = list(filter(lambda i: i not in sites_not_available, self.sites_available))
        
            # Sites are not available because a particle has migrated there
            for idx in sites_not_available:
                self.grid_crystal[idx].available_migrations(self.grid_crystal)
                self.grid_crystal[idx].transition_rates()

    def track_time(self,t):
        
        self.time += t



    def plot_lattice_points(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.position_cartesian[:,0]
        y = self.position_cartesian[:,1]
        z = self.position_cartesian[:,2]
        
        ax.scatter3D(x, y, z, c='blue', marker='o')
        ax.set_aspect('equal', 'box')
        
        ax.set_xlabel('x-axis (nm)')
        ax.set_ylabel('y-axis (nm)')
        ax.set_zlabel('z-axis (nm)')
        
        plt.show()
        
        
    def plot_crystal(self):
        
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        x = [site.position[0] for site in (self.grid_crystal[idx] for idx in self.sites_occupied)]
        y = [site.position[1] for site in (self.grid_crystal[idx] for idx in self.sites_occupied)]
        z = [site.position[2] for site in (self.grid_crystal[idx] for idx in self.sites_occupied)]
                
        ax.scatter3D(x, y, z, c='blue', marker='o')
        
        ax.set_xlabel('x-axis (nm)')
        ax.set_ylabel('y-axis (nm)')
        ax.set_zlabel('z-axis (nm)')
        
        ax.set_xlim([0, self.crystal_size[0]]) 
        ax.set_ylim([0, self.crystal_size[1]])
        ax.set_zlim([0, 2*self.crystal_size[2]])
        ax.set_aspect('equal', 'box')

        plt.show()
