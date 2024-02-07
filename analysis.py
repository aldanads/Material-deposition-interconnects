# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:31:14 2022

@author: ALDANADS
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


"""
    -Nucleation time?
"""
class Island:
    def __init__(self,z_starting_position,z_starting_pos_cart,island_sites):
        self.z_starting_position = z_starting_position
        self.z_starting_pos_cart = z_starting_pos_cart
        self.island_sites = island_sites
        

def calculate_mass(Co_latt):
    n_particles = len(Co_latt.sites_occupied)
    mass_specie = 63.546 # (mass of Copper in u) 
    x_size, y_size = Co_latt.crystal_size[:2]
    density = n_particles * mass_specie / (x_size * y_size)
    g_to_ng = 1e9
    nm_to_cm = 1e7

    return nm_to_cm**2 * g_to_ng * density / constants.Avogadro # (ng/cm2)


# =============================================================================
# We calculate % occupy per layer
# Average the contribution of each layer to the thickness acording to the z step
# =============================================================================
def average_thickness(Co_latt):
    
    grid_crystal = Co_latt.grid_crystal
    z_step = Co_latt.basis_vectors[0][2]
    z_steps = int(Co_latt.crystal_size[2]/z_step + 1)
    layers = [0] * z_steps  # Initialize each layer separately
    
    for site in grid_crystal.values():
        z_idx = int(round(site.position[2] / z_step))
        layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0
        
    sites_per_layer = len(grid_crystal)/z_steps
    normalized_layers = [count / sites_per_layer for count in layers]
    # Layer 0 is z = 0, so it doesn't contribute
    thickness = sum(normalized_layers[1:]) * z_step # (nm)    
        
    return thickness, normalized_layers,layers

def terrace_area(Co_latt,layers):
    
    grid_crystal = Co_latt.grid_crystal
    z_step = Co_latt.basis_vectors[0][2]
    z_steps = int(Co_latt.crystal_size[2]/z_step + 1)
    sites_per_layer = len(grid_crystal)/z_steps

    area_per_site = Co_latt.crystal_size[0] * Co_latt.crystal_size[1] / sites_per_layer
    
    terraces = [(layers[i-1] - layers[i]) * area_per_site for i in range(1,len(layers))]
    terraces.append(layers[-1] * area_per_site) # (nm2)
    
    return terraces
    
def RMS_roughness(z):
    
    z_mean = np.mean(z)
    return np.sqrt(np.mean((np.array(z)-z_mean)**2))

def plot_crystal_surface(Co_latt):
    
    grid_crystal = Co_latt.grid_crystal

    
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
            
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
    
    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_zlim([0, Co_latt.crystal_size[2]])
    
    ax.view_init(azim=45, elev = 45)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # Show the plot
    plt.show()
    return z
    
def detect_islands(grid_crystal,idx_site,visited,island_slice,chemical_specie):

    site = grid_crystal[idx_site] 
    
    if idx_site not in visited and site.chemical_specie == chemical_specie:
        visited.add(idx_site)
        island_slice.add(idx_site)
        # dfs_recursive
        for idx in site.migration_paths['Plane']:
            visited,island_slice = detect_islands(grid_crystal,idx[0],visited,island_slice,chemical_specie)
                                   
    return visited,island_slice

def build_island(grid_crystal,visited,island_sites,island_slice,chemical_specie):
    
    for site in island_slice:
        
        for element in grid_crystal[site].migration_paths['Up']:

            if element[0] not in visited and grid_crystal[element[0]].chemical_specie == chemical_specie:
                visited.add(element[0])
                island_sites.add(element[0])
                visited,island_sites = build_island(grid_crystal,visited,island_sites,island_slice,chemical_specie)
                
    return visited,island_sites
    
    
            


plt.rcParams["figure.dpi"] = 300
system = ['Windows','Linux']
choose_system = system[0]

if choose_system == 'Windows':
    import shelve
    
    filename = 'variables'
    
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
    
    
elif choose_system == 'Linux':
    
    import pickle
    filename = 'variables.pkl'
    
    # Open the file in binary mode
    with open(filename, 'rb') as file:
      
        # Call load method to deserialze
        myvar = pickle.load(file)
        
    Co_latt = myvar['Co_latt']


mass_gained = calculate_mass(Co_latt)
thickness, normalized_layers,layers = average_thickness(Co_latt)

terraces = terrace_area(Co_latt,layers)

fraction_sites_occupied = len(Co_latt.sites_occupied) / len(Co_latt.grid_crystal) 
z = plot_crystal_surface(Co_latt)
surf_roughness_RMS = RMS_roughness(z)

#Detect islands
grid_crystal = Co_latt.grid_crystal
chemical_specie = Co_latt.chemical_specie
visited = set()
sites_occupied = Co_latt.sites_occupied

count_islands = [0] * len(normalized_layers)
layers_no_complete = np.where(np.array(normalized_layers) != 1.0)
count_islands[normalized_layers == 1] = 1
z_step = Co_latt.basis_vectors[0][2]

islands_list = []

for z_idx in layers_no_complete[0]:    
    z_layer = round(z_idx * z_step,3)
    for idx_site in sites_occupied:     
        if grid_crystal[idx_site].position[2] == z_layer: 
            island_slice = set()
            visited,island_slice = detect_islands(grid_crystal,idx_site,visited,island_slice,chemical_specie)

            if len(island_slice):
                island_sites = island_slice.copy()
                visited,island_sites = build_island(grid_crystal,visited,island_sites,island_slice,chemical_specie)
                islands_list.append(Island(z_idx,z_layer,island_sites))
                count_islands[z_idx] += 1
