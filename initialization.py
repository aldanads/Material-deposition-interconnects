# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:03:03 2024

@author: samuel.delgado
"""
import numpy as np
import matplotlib.pyplot as plt
import platform
import shutil
import os 
from crystal_lattice import Crystal_Lattice

def initialization(n_sim,save_data):
    
    # Random seed as time
    rng = np.random.default_rng() # Random Number Generator (RNG) object

    # Default resolution for figures
    plt.rcParams["figure.dpi"] = 300 # Default value of dpi = 300
    
    if save_data:
        files_copy = ['initialization.py', 'crystal_lattice.py','Node.py','main.py','']
        
        if platform.system() == 'Windows': # When running in laptop
            dst = r'path\\'
        elif platform.system() == 'Linux': # HPC works on Linux
            dst = r'path/'
            
        paths = save_simulation(files_copy,dst,n_sim) # Create folders and python files
    else:
        paths = {'data': ''}
        
        
# =============================================================================
#         Experimental conditions
#         
# =============================================================================
    sticking_coeff = 1
    partial_pressure = 2 # (Pa = N m^-2 = kg m^-1 s^-2)
    mass_specie = 63.546 # (mass of Copper in u) 
    chemical_specie = 'Cu'
    T = 300 # (K)
    
    experimental_conditions = [sticking_coeff,partial_pressure,mass_specie,T,chemical_specie]
    
# =============================================================================
#         Crystal structure
#         
# =============================================================================
    a = 0.358 # (nm)
    b = 0.358 # (nm)
    c = 0.358 # (nm)
    lattice_constants = (a,b,c)
    crystal_size = (3, 3,1) # (nm)
    bravais_latt = ['fcc']
    orientation = ['001','111']
    lattice_properties = [lattice_constants,crystal_size,bravais_latt[0],orientation[1]]
    
# =============================================================================
#     Activation energies
#       - Migrating upward/downward one layer - It seems is promoted by other atoms surrounding
#       - Migrating upward/downward two layers in one jump
#           Nies, C. L., Natarajan, S. K., & Nolan, M. (2022). 
#           Control of the Cu morphology on Ru-passivated and Ru-doped TaN surfaces-promoting growth of 2D conducting copper for CMOS interconnects. 
#           Chemical Science, 13(3), 713–725. https://doi.org/10.1039/d1sc04708f
#       - Migrating in plane
#       - Clustering: Once they are together, difficult to split? 
#
# 
#       Jamnig, A., Sangiovanni, D. G., Abadias, G., & Sarakinos, K. (2019). 
#       Atomic-scale diffusion rates during growth of thin metal films on weakly-interacting substrates. 
#       Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-43107-8
#       - Migration of Cu on graphite - 0.05-0.13 eV
# =============================================================================
    select_dataset = 0    
    Act_E_dataset = ['TaN','Ru25','Ru50','test']   
    E_dataset = {'TaN':[0.4,0.7,0.33,0.84,0.44,0.76,0.74],
              'Ru25':[0.4,0.92,1.58,0.94,0.30,1.21,1.25],
              'Ru50':[0.4,0.62,0.78,1.18,1.08,1.86,1.82],
              'test':[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}
    E_mig_plane = E_dataset[Act_E_dataset[select_dataset]][0] # (eV)
    E_mig_upward_subs_layer1 = E_dataset[Act_E_dataset[select_dataset]][1]
    E_mig_downward_layer1_subs = E_dataset[Act_E_dataset[select_dataset]][2]
    E_mig_upward_layer1_layer2 = E_dataset[Act_E_dataset[select_dataset]][3]
    E_mig_upward_layer2_layer1 = E_dataset[Act_E_dataset[select_dataset]][4]
    E_mig_upward_subs_layer2 = E_dataset[Act_E_dataset[select_dataset]][5]
    E_mig_upward_layer2_subs = E_dataset[Act_E_dataset[select_dataset]][6]
    
    Act_E_list = [E_mig_plane,
                  E_mig_upward_subs_layer1,E_mig_downward_layer1_subs,
                  E_mig_upward_layer1_layer2,E_mig_upward_layer2_layer1,
                  E_mig_upward_subs_layer2,E_mig_upward_layer2_subs]


    # Binding energy | Desorption energy: https://doi.org/10.1039/D1SC04708F
    # Surface: [0]-TaN, [1]-Ru25, [2]-Ru50, [3]-Ru100, [4]-1 ML Ru passivation
    desorption_energy = {'TaN':3.49, 'Ru25':3.58, 'Ru50':3.59, 'Ru100':3.64, '1 ML Ru': 4.12}
    

# =============================================================================
#     Initialize the crystal grid structure - nodes with empty spaces
# =============================================================================
    Co_latt = Crystal_Lattice(lattice_properties,experimental_conditions,Act_E_list)
    
    
# =============================================================================
#     - test[0] - Normal deposition
#     - test[1] - Introduce a single particle in a determined site
#     - test[2] - Introduce and remove a single particle in a determined site: 
#     - test[3] - Hexagonal seed - 7 particles in plane
#     - test[4] - Hexagonal seed - 7 particles in plane and 1 on the top of the layer
#     - test[5] - 2 hexagonal seeds - 2 layers and one particle on the top 
# =============================================================================
    test = [0,1,2,3,4,5]

    # Deposition process of chemical species
    Co_latt.deposition_specie(0.0001,rng,test[5])


    return Co_latt,rng    
    
def save_simulation(files_copy,dst,n_sim):
    

    if platform.system() == 'Windows':
        parent_dir = 'Sim_'+str(n_sim)+'\\'
        os.makedirs(dst+parent_dir) 
        dst = dst+parent_dir
        program_directory = 'Program\\'
        data_directoy = 'Copper deposition\\'
        
    elif platform.system() == 'Linux':
        parent_dir = 'Sim_'+str(n_sim)+'/'
        os.makedirs(dst+parent_dir) 
        dst = dst+parent_dir
        program_directory = 'Program/'
        data_directoy = 'Copper deposition/'

    os.makedirs(dst + program_directory)
    os.makedirs(dst + data_directoy)
    
    paths = {'data': dst + data_directoy, 'program': dst + program_directory}

    for files in files_copy:
        shutil.copyfile(files, paths['program']+files)
        
    return paths
    