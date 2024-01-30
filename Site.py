# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:12 2024

@author: samuel.delgado
"""
from scipy import constants
import numpy as np

class Site():
    
    def __init__(self,chemical_specie,position,Act_E_list):
        
        self.chemical_specie = chemical_specie
        self.position = position
        self.nearest_neighbors_idx = [] # Nearest neighbors indexes
        self.nearest_neighbors_cart = [] # Nearest neighbors cartesian coordinates
        self.Act_E_list = Act_E_list
        self.site_events = [] # Possible events corresponding to this node
        self.migration_paths = {'Plane':[],'Up':[],'Down':[]} # Possible migration sites with the corresponding label
       
        # Set() to store the occupied sites that support this node
        # We start from zero everytime in case the 
        self.supp_by = set()
        # Position close to 0 are supported by the substrate
        if self.position[2] == 0:
            self.supp_by.add('Substrate')
        
            
# =============================================================================
#     We only consider the neighbors within the lattice domain            
# =============================================================================
    def neighbors_analysis(self,grid_crystal,neigh_idx,neigh_cart,crystal_size):
       
        self.num_mig_path = len(neigh_idx) + 2 # We consider two layers jumps: +1 upward, +1 downward
        num_event = 0
        for idx,pos in zip(neigh_idx,neigh_cart):
            if tuple(idx) in grid_crystal:
                self.nearest_neighbors_idx.append(tuple(idx))             
                self.nearest_neighbors_cart.append(tuple(pos))
                # Migration in the plane
                if round(pos[2]-self.position[2],3) == 0:
                    self.migration_paths['Plane'].append([tuple(idx),num_event,self.Act_E_list[0]])
                # Migration upward
                elif round(pos[2]-self.position[2],3) > 0:
                    # From substrate to layer 1
                    if 'Substrate' in self.supp_by:
                        self.migration_paths['Up'].append([tuple(idx),num_event,self.Act_E_list[1]])
                    # From layer_n to layer_n+1
                    else:
                        self.migration_paths['Up'].append([tuple(idx),num_event,self.Act_E_list[3]])
                # Migration downward
                elif round(pos[2]-self.position[2],3) < 0:
                    # From layer 1 to substrate
                    if 'Substrate' in grid_crystal[tuple(idx)].supp_by:
                        self.migration_paths['Down'].append([tuple(idx),num_event,self.Act_E_list[2]])
                    # From layer_n to layer_n-1
                    else:
                        self.migration_paths['Down'].append([tuple(idx),num_event,self.Act_E_list[4]])

                    
            # Establish boundary conditions for neighbors in xy plane
            # If pos is out of the boundary in xy but within z limits:
            elif (0 <= pos[2] <= crystal_size[2]):
                # Apply periodic boundary conditions in the xy plane
                pos = (round(pos[0] % crystal_size[0], 3), round(pos[1] % crystal_size[1], 3), pos[2])
    
                # Find the nearest neighbor within the grid
                min_dist, min_dist_idx = min(
                    ((np.linalg.norm(np.array(site.position) - np.array(pos)), idx) for idx, site in grid_crystal.items()),
                    key=lambda x: x[0]
                )
    
                self.nearest_neighbors_idx.append(tuple(min_dist_idx))
                self.nearest_neighbors_cart.append(tuple(grid_crystal[min_dist_idx].position))
                # Migration in the plane
                if round(pos[2]-self.position[2],3) == 0:
                    self.migration_paths['Plane'].append([tuple(min_dist_idx),num_event,self.Act_E_list[0]])
                # Migration upward
                elif round(pos[2]-self.position[2],3) > 0:
                    # From substrate to layer 1
                    if 'Substrate' in self.supp_by:
                        self.migration_paths['Up'].append([tuple(min_dist_idx),num_event,self.Act_E_list[1]])
                    # From layer_n to layer_n+1
                    else:
                        self.migration_paths['Up'].append([tuple(min_dist_idx),num_event,self.Act_E_list[3]])
                # Migration downward
                elif round(pos[2]-self.position[2],3) < 0:
                    # From layer 1 to substrate
                    if 'Substrate' in grid_crystal[tuple(min_dist_idx)].supp_by:
                        self.migration_paths['Down'].append([tuple(min_dist_idx),num_event,self.Act_E_list[2]])
                    # From layer_n to layer_n-1
                    else:
                        self.migration_paths['Down'].append([tuple(min_dist_idx),num_event,self.Act_E_list[4]])
              
                
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
                
        self.calculate_clustering_energy()
                
    def calculate_clustering_energy(self):
        
        # If this site is supported by the substrate, we add the binding energy to the substrate
        # We reduce 1 if it is supported by the substrate
        # We add 1 because if the site is occupied
        if 'Substrate' in self.supp_by and self.chemical_specie != 'Empty':
            self.energy_site = self.Act_E_list[-1][len(self.supp_by)] + self.Act_E_list[-2]
        elif 'Substrate' in self.supp_by and self.chemical_specie == 'Empty':
            self.energy_site = self.Act_E_list[-1][len(self.supp_by)-1] + self.Act_E_list[-2]
        elif 'Substrate' not in self.supp_by and self.chemical_specie != 'Empty':
            self.energy_site = self.Act_E_list[-1][len(self.supp_by)+1]
        elif 'Substrate' not in self.supp_by and self.chemical_specie == 'Empty':
            self.energy_site = self.Act_E_list[-1][len(self.supp_by)]
   
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
      
# =============================================================================
#         Lü, B., Almyras, G. A., Gervilla, V., Greene, J. E., & Sarakinos, K. (2018). 
#         Formation and morphological evolution of self-similar 3D nanostructures on weakly interacting substrates. 
#         Physical Review Materials, 2(6). https://doi.org/10.1103/PhysRevMaterials.2.063401
#         - Number of nearest neighbors needed to support a site so a particle can migrate there
# =============================================================================
        new_site_events = []

        # Plane migrations
        for site_idx, num_event, act_energy in self.migration_paths['Plane']:
            if site_idx not in self.supp_by and ('Substrate' in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 2):
                energy_change = max(grid_crystal[site_idx].energy_site - self.energy_site, 0)
                new_site_events.append([site_idx, num_event, act_energy + energy_change])

# =============================================================================
#         Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2020). 
#         The role of Ru passivation and doping on the barrier and seed layer properties of Ru-modified TaN for copper interconnects. 
#         Journal of Chemical Physics, 152(14). https://doi.org/10.1063/5.0003852
#   
#         - Migration upward stable is supported by three particles??  
# =============================================================================                      
        # Upward migrations
        for site_idx, num_event, act_energy in self.migration_paths['Up']:
            # Supported by at least 2 particles (excluding this site)
            if site_idx not in self.supp_by and len(grid_crystal[site_idx].supp_by) > 2:
                energy_change = max(grid_crystal[site_idx].energy_site - self.energy_site, 0)
                new_site_events.append([site_idx, num_event, act_energy + energy_change])
                
                # 2 layers jump upward
                for next_neighbor in grid_crystal[site_idx].migration_paths['Up']:
                    # Supported by at least 2 particles (this site is too far)
                    if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and len(grid_crystal[next_neighbor[0]].supp_by) > 1:
                        energy_change = max(grid_crystal[next_neighbor[0]].energy_site - self.energy_site, 0)
                        new_site_events.append([next_neighbor[0], self.num_event + 1, self.Act_E_list[5] + energy_change])


        # Downward migrations
        for site_idx, num_event, act_energy in self.migration_paths['Down']:
            # Supported by at least 2 particles (excluding this site)
            if site_idx not in self.supp_by and ('Substrate' in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 2):
                energy_change = max(grid_crystal[site_idx].energy_site - self.energy_site, 0)
                new_site_events.append([site_idx, num_event, act_energy + energy_change])
                
                # 2 layers jump downward
                for next_neighbor in grid_crystal[site_idx].migration_paths['Down']:
                    # Supported by at least 2 particles (this site is too far)
                    if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and (('Substrate' in grid_crystal[next_neighbor[0]].supp_by) or len(grid_crystal[next_neighbor[0]].supp_by) > 1):
                        energy_change = max(grid_crystal[next_neighbor[0]].energy_site - self.energy_site, 0)
                        new_site_events.append([next_neighbor[0], self.num_event + 2, self.Act_E_list[6] + energy_change])

            
        self.site_events = new_site_events
      

# =============================================================================
#         Calculate transition rates    
# =============================================================================
    def transition_rates(self,T = 300):
        
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E12;  # nu0 (s^-1) bond vibration frequency
        
        TR = [nu0*np.exp(-event[-1]/(kb*T)) for event in self.site_events]
                
        # Iterate over site_events directly, no need to use range(len(...))
        for event, tr_value in zip(self.site_events, TR):
            # Use the length of event to determine the appropriate action
            if len(event) == 3:
                # Insert at the beginning of the list for the binary tree
                event.insert(0, tr_value)
            elif len(event) == 4:
                event[0] = tr_value
                
        
