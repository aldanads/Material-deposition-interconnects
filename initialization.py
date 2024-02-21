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
    plt.rcParams["figure.dpi"] = 100 # Default value of dpi = 300
    
    if save_data:
        files_copy = ['initialization.py', 'crystal_lattice.py','Site.py','main.py','KMC.py','balanced_tree.py','analysis.py']
        
        if platform.system() == 'Windows': # When running in laptop
            dst = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Test_cluster\1st nearest neighbors_2\\'
        elif platform.system() == 'Linux': # HPC works on Linux
            dst = r'/sfiwork/samuel.delgado/Copper_deposition/test/'
            
        paths,Results = save_simulation(files_copy,dst,n_sim) # Create folders and python files
        
    else:
        paths = {'data': ''}
        Results = []
        
        
# =============================================================================
#         Experimental conditions
#         
# =============================================================================
    sticking_coeff = 1
    partial_pressure = 5 # (Pa = N m^-2 = kg m^-1 s^-2)
    mass_specie = 63.546 # (mass of Copper in u) 
    chemical_specie = 'Cu'
    T = 573 # (K)
    
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
#       Different surface Structures- fcc Metals
#       https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Surface_Science_(Nix)/01%3A_Structure_of_Solid_Surfaces/1.03%3A_Surface_Structures-_fcc_Metals
#       Activation energies
#       Nies, C. L., Natarajan, S. K., & Nolan, M. (2022). 
#       Control of the Cu morphology on Ru-passivated and Ru-doped TaN surfaces-promoting growth of 2D conducting copper for CMOS interconnects. 
#       Chemical Science, 13(3), 713–725. https://doi.org/10.1039/d1sc04708f
#           - Migrating upward/downward one layer - It seems is promoted by other atoms surrounding
#           - Migrating upward/downward two layers in one jump
# 
#       Jamnig, A., Sangiovanni, D. G., Abadias, G., & Sarakinos, K. (2019). 
#       Atomic-scale diffusion rates during growth of thin metal films on weakly-interacting substrates. 
#       Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-43107-8
#           - Migration of Cu on graphite - 0.05-0.13 eV
# 
#       Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2019). 
#       Ru passivated and Ru doped ϵ-TaN surfaces as a combined barrier and liner material for copper interconnects: A first principles study. 
#       Journal of Materials Chemistry C, 7(26), 7959–7973. https://doi.org/10.1039/c8tc06118a
#           - TaN (111) - Activation energy for Cu migration - [0.85 - 1.26] (ev)
#           - Ru(0 0 1) - Activation energy for Cu migration - [0.07 - 0.11] (ev)
#           - 1ML Ru - Activation energy for Cu migration - [0.01, 0.21, 0.45, 0.37] (ev)
#           - 2ML Ru - Activation energy for Cu migration - [0.46, 0.44] (ev)
#           - Information about clustering two Cu atoms on TaN and Ru surfaces
# =============================================================================
    select_dataset = 3   
    Act_E_dataset = ['TaN','Ru25','Ru50','test']  

    E_dataset = {'TaN':[0.85,0.7,0.33,0.84,0.44,0.76,0.74],
              'Ru25':[0.4,0.92,1.58,0.94,0.30,1.21,1.25],
              'Ru50':[0.4,0.62,0.78,1.18,1.08,1.86,1.82],
               'test':[0.85,0.21,0.322,0.322,0.322,0.76,0.74]}
             
# =============================================================================
#     Böyükata, M., & Belchior, J. C. (2008). 
#     Structural and Energetic Analysis of Copper Clusters: MD Study of Cu n (n = 2-45). 
#     In J. Braz. Chem. Soc (Vol. 19, Issue 5).
#      - Clustering energy
# 
#     Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2020). 
#     The role of Ru passivation and doping on the barrier and seed layer properties of Ru-modified TaN for copper interconnects. 
#     Journal of Chemical Physics, 152(14). https://doi.org/10.1063/5.0003852
#       
#       They have calculated for 2-4 atoms, the rest is extrapolation
#       It is in (eV/Cu)
#       'TaN': [0,0,-1,-1.2,-1.34,-1.46,-1.58,-1.7,-1.82,-1.94,-2.06,-2.18,-2.3,-2.42]
#       'Ru25':[0,0,-0.55,-0.4,-1.30,-1.42,-1.54,-1.66,-1.78,-1.9,-2.02,-2.14,-2.26,-2.38]
#       'Ru50':[0,0,-1,-1.22,-1.34,-1.46,-1.58,-1.7,-1.82,-1.94,-2.06,-2.18,-2.3,-2.42]}
# =============================================================================    
    E_clustering = {'Void': [0,0,-0.577,-1.732,-3.465,-5.281,-7.566,-9.676,-11.902,-14.228,-16.848,-19.643,-22.818,-26.710],
                    'TaN': [0,0,-1 * 2,-1.2 * 3,-1.34 * 4,-1.46 * 5,-1.58 * 6,-1.7 * 7,-1.82 * 8,-1.94 * 9,-2.06 * 10,-2.18 * 11,-2.3 * 12,-2.42 * 13],
                    'Ru25':[0,0,-0.55 * 2,-0.4 * 3,-1.30 * 4,-1.42 * 5,-1.54 * 6,-1.66 * 7,-1.78 * 8,-1.9 * 9,-2.02 * 10,-2.14 * 11,-2.26 * 12,-2.38 * 13],
                    'Ru50':[0,0,-1 * 2,-1.22 * 3,-1.34 * 4,-1.46 * 5,-1.58 * 6,-1.7 * 7,-1.82 * 8,-1.94 * 9,-2.06 * 10,-2.18 * 11,-2.3 * 12,-2.42 * 13],
                    'test':[0,0,-0.252 * 2,-0.252 * 3,-0.252 * 4,-0.252 * 5,-0.252 * 6,-0.252 * 7,-0.252 * 8,-0.252 * 9,-0.252 * 10,-0.252 * 11,-0.252 * 12,-0.252 * 13]} 

    
    E_mig_plane_sub = E_dataset[Act_E_dataset[select_dataset]][0] # (eV)
    E_mig_upward_subs_layer1 = E_dataset[Act_E_dataset[select_dataset]][1]
    E_mig_downward_layer1_subs = E_dataset[Act_E_dataset[select_dataset]][2]
    E_mig_upward_layer1_layer2 = E_dataset[Act_E_dataset[select_dataset]][3]
    E_mig_downward_layer2_layer1 = E_dataset[Act_E_dataset[select_dataset]][4]
    E_mig_upward_subs_layer2 = E_dataset[Act_E_dataset[select_dataset]][5]
    E_mig_downward_layer2_subs = E_dataset[Act_E_dataset[select_dataset]][6]
# =============================================================================
#     Papanicolaou, N. 1, & Evangelakis, G. A. (n.d.). 
#     COMPARISON OF DIFFUSION PROCESSES OF Cu AND Au ADA TOMS ON THE Cu(1l1) SURFACE BY MOLECULAR DYNAMICS.
#     
#     Mińkowski, Marcin, and Magdalena A. Załuska-Kotur. 
#     "Diffusion of Cu adatoms and dimers on Cu (111) and Ag (111) surfaces." 
#     Surface Science 642 (2015): 22-32. 10.1016/j.susc.2015.07.026
# =============================================================================
    #E_mig_plane_Cu = 0.05*(n_sim+1) # (eV)
    E_mig_plane_Cu = 0.081 # (eV)

    # Binding energy | Desorption energy: https://doi.org/10.1039/D1SC04708F
    # Surface: [0]-TaN, [1]-Ru25, [2]-Ru50, [3]-Ru100, [4]-1 ML Ru passivation
    binding_energy = {'TaN':-3.49, 'Ru25':-3.58, 'Ru50':-3.59, 'Ru100':-3.64, '1 ML Ru':-4.12, 'test':-0.1 * n_sim}
    Act_E_list = [E_mig_plane_sub,
                  E_mig_upward_subs_layer1,E_mig_downward_layer1_subs,
                  E_mig_upward_layer1_layer2,E_mig_downward_layer2_layer1,
                  E_mig_upward_subs_layer2,E_mig_downward_layer2_subs,
                  E_mig_plane_Cu,
                  binding_energy['test'],E_clustering[Act_E_dataset[select_dataset]]]

# =============================================================================
#     Initialize the crystal grid structure - nodes with empty spaces
# =============================================================================
    Co_latt = Crystal_Lattice(lattice_properties,experimental_conditions,Act_E_list)
 
    # Maximum probability per site for deposition to establish a timestep limits
    # The maximum timestep is that one that occupy 10% of the site during the deposition process
    P_limits = 0.1
    Co_latt.limit_kmc_timestep(P_limits)
    
# =============================================================================
#     - test[0] - Normal deposition
#     - test[1] - Introduce a single particle in a determined site
#     - test[2] - Introduce and remove a single particle in a determined site 
#     - test[3] - Introduce two adjacent particles
#     - test[4] - Hexagonal seed - 7 particles in plane + 1 particle in plane
#     - test[5] - Hexagonal seed - 7 particles in plane and 1 on the top of the layer
#     - test[6] - 2 hexagonal seeds - 2 layers and one particle on the top 
# =============================================================================
    test = [0,1,2,3,4,5,6,7,8]

    # Deposition process of chemical species
    Co_latt.deposition_specie(Co_latt.timestep_limits,rng,test[8])
    Co_latt.track_time(Co_latt.timestep_limits) 
    Co_latt.add_time()

    return Co_latt,rng,paths,Results
    
def save_simulation(files_copy,dst,n_sim):
    

    if platform.system() == 'Windows':
        parent_dir = 'Sim_'+str(n_sim)+'\\'
        os.makedirs(dst+parent_dir) 
        dst = dst+parent_dir
        program_directory = 'Program\\'
        data_directoy = 'Crystal evolution\\'
        current_directory = os.path.dirname(__file__)

        
    elif platform.system() == 'Linux':
        parent_dir = 'Sim_'+str(n_sim)+'/'
        os.makedirs(dst+parent_dir) 
        dst = dst+parent_dir
        program_directory = 'Program/'
        data_directoy = 'Crystal evolution/'
        current_directory = os.path.dirname(__file__)

    os.makedirs(dst + program_directory)
    os.makedirs(dst + data_directoy)
    
    paths = {'data': dst + data_directoy, 'program': dst + program_directory,'results': dst}

    for file in files_copy:
        # Utilizing os.path.join is the best option, because it works in Windows and Unix
        # and put the correct separators
        source_file = os.path.join(current_directory, file)
        destination_file = os.path.join(paths['program'], file)
        shutil.copyfile(source_file, destination_file)
        
    excel_filename = dst + 'Results.csv'
    Results = SimulationResults(excel_filename)
        
    return paths,Results

def save_variables(paths,variables):
    
    
    if platform.system() == 'Windows': # When running in laptop

        import shelve
    
        filename = 'variables'
        my_shelf = shelve.open(paths+filename,'n') # 'n' for new
        
        for key in variables:
            my_shelf[key] = variables[key]
    
        my_shelf.close()

    elif platform.system() == 'Linux': # HPC works on Linux
    
        import pickle
    
        filename = 'variables.pkl'    
    
        # Open a file and use dump()
        with open(paths+filename, 'wb') as file:
              
            # A new file will be created
            pickle.dump(variables,file)
            
class SimulationResults:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        # Initialize a CSV file with headers
        with open(excel_filename, 'w') as f:
            f.write('Time,Mass,Sites Occupation,Average Thickness,Terrace Area,RMS Roughness,Performance time\n')
    
    def measurements_crystal(self, time, mass_gained, sites_occupation, thickness, avg_terrace, surf_roughness_RMS,performance_time):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{time},{mass_gained},{sites_occupation},{thickness},{avg_terrace},{surf_roughness_RMS},{performance_time}\n')