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
    partial_pressure = 1 # (Pa = N m^-2 = kg m^-1 s^-2)
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
    crystal_size = (3, 3,1)
    bravais_latt = ['fcc']
    orientation = ['001','111']
    lattice_properties = [lattice_constants,crystal_size,bravais_latt[0],orientation[1]]
    
# =============================================================================
#     Activation energies
#     
# =============================================================================
    E_mig = 0.4 # (eV)

    # Binding energy | Desorption energy: https://doi.org/10.1039/D1SC04708F
    # Surface: [0]-TaN, [1]-Ru25, [2]-Ru50, [3]-Ru100, [4]-1 ML Ru passivation
    desorption_energy = [3.49, 3.58, 3.59, 3.64, 4.12]

# =============================================================================
#     Initialize the crystal grid structure - nodes with empty spaces
# =============================================================================
    Co_latt = Crystal_Lattice(lattice_properties,experimental_conditions,E_mig)
    
    
# =============================================================================
#     - test[0] - Normal deposition
#     - test[1] - Introduce a single particle in a determined site
#     - test[2] - Introduce and remove a single particle in a determined site
# =============================================================================
    test = [0,1,2]

    # Deposition process of chemical species
    Co_latt.deposition_specie(0.0001,rng,test[0])


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
    