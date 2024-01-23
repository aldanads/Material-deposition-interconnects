# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:12 2024

@author: samuel.delgado
"""
from scipy import constants
import numpy as np

class Site():
    
    def __init__(self,chemical_specie,position,activation_energies):
        
        self.chemical_specie = chemical_specie
        self.position = position
        self.nearest_neighbors_idx = [] # Nearest neighbors indexes
        self.nearest_neighbors_cart = [] # Nearest neighbors cartesian coordinates
        self.activation_energies = activation_energies
        self.site_events = [] # Possible events corresponding to this node
        self.migration_paths = [] # Possible migration sites with the corresponding label
       
        # Set() to store the occupied sites that support this node
        # We start from zero everytime in case the 
        self.supp_by = set()
        # Position close to 0 are supported by the substrate
        if self.position[2] < 1e-16:
            self.supp_by.add('Substrate')
        
            
# =============================================================================
#     We only consider the neighbors within the lattice domain            
# =============================================================================
    def neighbors_analysis(self,grid_crystal,neigh_idx,neigh_cart):
       
        self.num_mig_path = len(neigh_idx)
        num_event = 0
        for idx,pos in zip(neigh_idx,neigh_cart):
            if tuple(idx) in grid_crystal:
                self.nearest_neighbors_idx.append(tuple(idx))             
                self.nearest_neighbors_cart.append(tuple(pos))
                self.migration_paths.append([tuple(idx),num_event])
            num_event+= 1
        self.num_event = num_event
            
# =============================================================================
#         Occupied sites supporting this node
# =============================================================================    
    def supported_by(self,grid_crystal):
                 
        # Go over the nearest neighbors
        for idx in self.nearest_neighbors_idx:
            idx = tuple(idx)
            # Select the occupied sites that support this node
            ## I don't need to check if idx is in the domain, as the method
            ## neighbors_analysis() select the neighbors within the domain
            if (grid_crystal[idx].chemical_specie != "Empty"):
                self.supp_by.add(idx)
            # If some of the sites that supported this node disappear, remove
            # it from the supp_by set()
            elif (grid_crystal[idx].chemical_specie == "Empty") and (idx in self.supp_by):
                self.supp_by.remove(idx)
   
# =============================================================================
#       Calculate the possible events corresponding to this node
#       - Migration events
#       - Desorption events
# =============================================================================
    # Change chemical_specie status
    # Add the desorption process
    def introduce_specie(self,chemical_specie):
        self.chemical_specie = chemical_specie
        #self.site_events.append(['Desorption',self.num_event])

    def remove_specie(self):
        self.chemical_specie = 'Empty'
        #self.site_events.remove(['Desorption',self.num_event])
        self.site_events = []

    # Calculate posible migration sites
    def available_migrations(self,grid_crystal):
        # Remove the sites that are already occupied - can't migrate to that site
        # self.site_events = list(filter(lambda x: x[0] != self.supp_by,
        #                                self.migration_paths)) + self.site_events
        
        for item in self.migration_paths:

            # It should be supported by more than one, that is, not only by the migrating particle
            if (item[0] not in self.supp_by) and (len(grid_crystal[item[0]].supp_by) > 1):
                # It should be a copy of item to not modify item in place -->
                # That modify migration_paths when we modify site_events
                self.site_events.append(item.copy())
                
        
        """
        The migration site should be supported by at least one specie
        """

 
    
# =============================================================================
#         Calculate transition rates    
# =============================================================================
    def transition_rates(self,T = 300):
        
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E13;  # nu0 (s^-1) bond vibration frequency
        TR = (nu0*np.exp(-np.array(self.activation_energies)/(kb*T)))
        
        for i in range(len(self.site_events)):
            
            if len(self.site_events[i]) == 2:
                # Insert at the beginning of the list for the binary tree
                self.site_events[i].insert(0,TR)
            elif len(self.site_events[i]) == 3:
                self.site_events[i][0] = TR
            
        
        
