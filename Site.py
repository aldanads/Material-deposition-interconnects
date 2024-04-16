# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:12 2024

@author: samuel.delgado
"""
from scipy import constants
import numpy as np
from sklearn.decomposition import PCA


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
                    self.migration_paths['Plane'].append([tuple(idx),num_event])

                # Migration upward
                elif round(pos[2]-self.position[2],3) > 0:
                    self.migration_paths['Up'].append([tuple(idx),num_event])
                    
                # Migration downward
                elif round(pos[2]-self.position[2],3) < 0:
                    self.migration_paths['Down'].append([tuple(idx),num_event])

                    
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
                    self.migration_paths['Plane'].append([tuple(min_dist_idx),num_event])
                    
                # Migration upward
                elif round(pos[2]-self.position[2],3) > 0:
                    self.migration_paths['Up'].append([tuple(min_dist_idx),num_event])
                    
                # Migration downward
                elif round(pos[2]-self.position[2],3) < 0:
                    self.migration_paths['Down'].append([tuple(min_dist_idx),num_event])
              
                
            num_event+= 1
        self.num_event = num_event
            
# =============================================================================
#         Occupied sites supporting this node
# =============================================================================    
    def supported_by(self,grid_crystal,crystallographic_planes):
                 
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
                

        self.detect_edges(grid_crystal)               
        self.calculate_clustering_energy()
        self.detect_planes(grid_crystal,crystallographic_planes)
                
    def calculate_clustering_energy(self,supp_by_destiny = 0,idx_origin = 0):
        
        # If this site is supported by the substrate, we add the binding energy to the substrate
        # We reduce 1 if it is supported by the substrate
        # We add 1 because if the site is occupied
        
        if supp_by_destiny == 0:
            if 'Substrate' in self.supp_by:
                self.energy_site = self.Act_E_list[-1][len(self.supp_by)] + self.Act_E_list[-2]
            else:
                self.energy_site = self.Act_E_list[-1][len(self.supp_by)+1]
                
        # We should consider the particle that would migrate there to calculate
        # the energy difference with the origin site
        else:
            if 'Substrate' in supp_by_destiny and idx_origin in supp_by_destiny:
                energy_site = self.Act_E_list[-1][len(supp_by_destiny)-1] + self.Act_E_list[-2]
            elif 'Substrate' in supp_by_destiny and idx_origin not in supp_by_destiny:
                energy_site = self.Act_E_list[-1][len(supp_by_destiny)] + self.Act_E_list[-2]
            elif 'Substrate' not in supp_by_destiny and idx_origin in supp_by_destiny:
                energy_site = self.Act_E_list[-1][len(supp_by_destiny)]
            elif 'Substrate' not in supp_by_destiny and idx_origin not in supp_by_destiny:
                energy_site = self.Act_E_list[-1][len(supp_by_destiny)+1]
            
            return energy_site
   

   
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
    def available_migrations(self,grid_crystal,idx_origin):
      
# =============================================================================
#         Lü, B., Almyras, G. A., Gervilla, V., Greene, J. E., & Sarakinos, K. (2018). 
#         Formation and morphological evolution of self-similar 3D nanostructures on weakly interacting substrates. 
#         Physical Review Materials, 2(6). https://doi.org/10.1103/PhysRevMaterials.2.063401
#         - Number of nearest neighbors needed to support a site so a particle can migrate there
# =============================================================================
        new_site_events = []

        # Plane migrations
        for site_idx, num_event in self.migration_paths['Plane']:
            if site_idx not in self.supp_by and ('Substrate' in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 2):
                energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                energy_change = max(energy_site_destiny - self.energy_site, 0)
                
                # Migrating on the substrate
                if 'Substrate' in self.supp_by:
                    new_site_events.append([site_idx, num_event, self.Act_E_list[0] + energy_change])
                    
                # Migrating on the film (111)
                elif grid_crystal[site_idx].crystallographic_direction == (111):
                    
                    if self.edges_v[num_event] == None: 
                        new_site_events.append([site_idx, num_event, self.Act_E_list[7] + energy_change])
                    elif self.edges_v[num_event] == 111:
                        new_site_events.append([site_idx, num_event, self.Act_E_list[10] + energy_change])
                    elif self.edges_v[num_event] == 100:
                        new_site_events.append([site_idx, num_event, self.Act_E_list[9] + energy_change])
                        
                # Migrating on the film (100)
                elif grid_crystal[site_idx].crystallographic_direction == (100):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])


# =============================================================================
#         Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2020). 
#         The role of Ru passivation and doping on the barrier and seed layer properties of Ru-modified TaN for copper interconnects. 
#         Journal of Chemical Physics, 152(14). https://doi.org/10.1063/5.0003852
#   
#         - Migration upward stable is supported by three particles??  
# =============================================================================                      
        # Upward migrations
        for site_idx, num_event in self.migration_paths['Up']:
        
# =============================================================================
#             """
#             Same activation energy and num_event than normal upward migrations
#             """
#             VERY EXPENSIVE to include 2nd nearest neighbors in the migration
# 
#             # Second nearest neighbors: 1 jump upward + 1 jump in plane --> Facilitate migration between layers without crossing the edge (only two neighbors supporting)
#             for next_neighbor in grid_crystal[site_idx].migration_paths['Plane']:
#                 # Supported by at least 2 particles (this site is far)
#                 if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and len(grid_crystal[next_neighbor[0]].supp_by) > 1:
#                     energy_site_destiny = self.calculate_clustering_energy(grid_crystal[next_neighbor[0]].supp_by,idx_origin)
#                     energy_change = max(energy_site_destiny - self.energy_site, 0)
#                     new_site_events.append([next_neighbor[0], num_event, act_energy + energy_change])
# 
# =============================================================================

                # First nearest neighbors: 1 jump upward
                # Supported by at least 2 particles (excluding this site)
            if site_idx not in self.supp_by and len(grid_crystal[site_idx].supp_by) > 2:
                energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                energy_change = max(energy_site_destiny - self.energy_site, 0)
                
                # Migrating upward from the substrate
                if 'Substrate' in self.supp_by and grid_crystal[site_idx].crystallographic_direction == (111):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[1] + energy_change])
                
                elif 'Substrate' in self.supp_by and grid_crystal[site_idx].crystallographic_direction == (100):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[5] + energy_change])
                    
                # Migrating upward from the film (111)
                elif self.crystallographic_direction == (111):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[3] + energy_change])
                    
                # Migrating upward from the film (100)
                elif self.crystallographic_direction ==  (100):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])


# =============================================================================
#             VERY EXPENSIVE to include 2nd nearest neighbors in the migration
#                 # Second nearest neighbors: 2 layers jump upward
#                 for next_neighbor in grid_crystal[site_idx].migration_paths['Up']:
#                     # Supported by at least 2 particles (this site is far)
#                     if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and len(grid_crystal[next_neighbor[0]].supp_by) > 1:
#                         energy_site_destiny = self.calculate_clustering_energy(grid_crystal[next_neighbor[0]].supp_by,idx_origin)
#                         energy_change = max(energy_site_destiny - self.energy_site, 0)
#                         new_site_events.append([next_neighbor[0], self.num_event + 1, self.Act_E_list[5] + energy_change])
#                 
# =============================================================================
                
        # Downward migrations
        for site_idx, num_event in self.migration_paths['Down']:
            
# =============================================================================
#             """
#             Same activation energy and num_event than normal downward migrations
#             """
#            VERY EXPENSIVE to include 2nd nearest neighbors in the migration

#             # Second nearest neighbors: 1 jump downward + 1 jump in plane --> Facilitate migration between layers without crossing the edge (only two neighbors supporting)
#             for next_neighbor in grid_crystal[site_idx].migration_paths['Plane']:
#                 # Supported by at least 2 particles (this site is too far)
#                 if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and (('Substrate' in grid_crystal[next_neighbor[0]].supp_by) or len(grid_crystal[next_neighbor[0]].supp_by) > 1):
#                     energy_site_destiny = self.calculate_clustering_energy(grid_crystal[next_neighbor[0]].supp_by,idx_origin)
#                     energy_change = max(energy_site_destiny - self.energy_site, 0)
#                     new_site_events.append([next_neighbor[0], num_event, act_energy + energy_change])
# =============================================================================

                # First nearest neighbors: 1 jump downward
                # Supported by at least 2 particles (excluding this site)
            if site_idx not in self.supp_by and ('Substrate' in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 2):
                energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                energy_change = max(energy_site_destiny - self.energy_site, 0)
                
                # From layer 1 to substrate
                if self.crystallographic_direction == (111) and 'Substrate' in grid_crystal[site_idx].supp_by:
                    new_site_events.append([site_idx, num_event, self.Act_E_list[2] + energy_change])
                
                elif self.crystallographic_direction == (100) and 'Substrate' in grid_crystal[site_idx].supp_by:
                    new_site_events.append([site_idx, num_event, self.Act_E_list[6] + energy_change])
                
                # Migrating downward from the film (111)
                elif self.crystallographic_direction == (111):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[4] + energy_change])
                    
                # Migrating downward from the film (100)
                elif self.crystallographic_direction == (100):
                    new_site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])
                
# =============================================================================
#             VERY EXPENSIVE to include 2nd nearest neighbors in the migration
#                 # Second nearest neighbors: 2 layers jump downward
#                 for next_neighbor in grid_crystal[site_idx].migration_paths['Down']:
#                     # Supported by at least 2 particles (this site is too far)
#                     if grid_crystal[next_neighbor[0]].chemical_specie == 'Empty' and (('Substrate' in grid_crystal[next_neighbor[0]].supp_by) or len(grid_crystal[next_neighbor[0]].supp_by) > 1):
#                         energy_site_destiny = self.calculate_clustering_energy(grid_crystal[next_neighbor[0]].supp_by,idx_origin)
#                         energy_change = max(energy_site_destiny - self.energy_site, 0)
#                         new_site_events.append([next_neighbor[0], self.num_event + 2, self.Act_E_list[6] + energy_change])
# 
# =============================================================================
            
        self.site_events = new_site_events
      
# =============================================================================
#     Detect planes using PCA - We search the plane that contains most of the points
#     in supp_by  to know the surface where this site is attached 
# =============================================================================
    def detect_planes(self,grid_crystal,crystallographic_planes):
        
        atom_coordinates = np.array([grid_crystal[idx].position for idx in self.supp_by if idx != 'Substrate'])

        if 'Substrate' in self.supp_by:
            self.crystallographic_direction = (111)
        elif len(atom_coordinates) > 2:
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(atom_coordinates)
            
            # Get the eigenvectors and eigenvalues
            eigenvectors = pca.components_
            eigenvalues = pca.explained_variance_
            
            # Sort eigenvectors based on eigenvalues
            sorted_indices = np.argsort(eigenvalues)[::-1]
            principal_components = eigenvectors[sorted_indices]
            
            # Define the plane by the two principal components with the largest eigenvalues
            plane_normal = np.cross(principal_components[0], principal_components[1])
            
            aux_min = 2
            for plane in crystallographic_planes:
                cross_product = np.cross(plane[1],plane_normal)
                norm_cross_product = np.linalg.norm(cross_product)
                if norm_cross_product < aux_min:
                    aux_min = norm_cross_product
                    self.crystallographic_direction = plane[0]
                    
        else:
            self.crystallographic_direction = (111)
            
    def detect_edges(self,grid_crystal):
        
        
        mig_paths = self.migration_paths['Plane']
        edges_v = {2:None, 3:None, 4:None, 6:None, 10:None, 11:None}
        
        for site_idx, num_event in self.migration_paths['Plane']:
            
            if num_event == 2 or num_event == 4:
                if (grid_crystal[mig_paths[3][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[4][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 100
                    
                if (grid_crystal[mig_paths[1][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[5][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 111
                
            elif num_event == 3 or num_event == 6:
                if (grid_crystal[mig_paths[0][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[4][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 111
                    
                if (grid_crystal[mig_paths[2][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[5][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 100 

            elif num_event == 10 or num_event == 11:
                if (grid_crystal[mig_paths[0][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[1][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 100 
                    
                if (grid_crystal[mig_paths[2][0]].chemical_specie == self.chemical_specie 
                    and grid_crystal[mig_paths[3][0]].chemical_specie == self.chemical_specie):
                    edges_v[num_event] = 111
                    
        self.edges_v = edges_v

                
            

            
            

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
                
        
class Island():
    def __init__(self,z_starting_position,z_starting_pos_cart,island_sites):
        self.z_starting_position = z_starting_position
        self.z_starting_pos_cart = z_starting_pos_cart
        self.island_sites = island_sites
