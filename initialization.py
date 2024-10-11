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
from superbasin import Superbasin
import copy


def initialization(n_sim,save_data):
    
    seed = 1
    # Random seed as time
    rng = np.random.default_rng(seed) # Random Number Generator (RNG) object

    # Default resolution for figures
    plt.rcParams["figure.dpi"] = 100 # Default value of dpi = 300
    
    if save_data:
        files_copy = ['initialization.py', 'crystal_lattice.py','Site.py','main.py','KMC.py','balanced_tree.py','analysis.py']
        
        if platform.system() == 'Windows': # When running in laptop
            dst = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Test\\'
        elif platform.system() == 'Linux': # HPC works on Linux
            dst = r'/sfiwork/samuel.delgado/Copper_deposition/Varying_substrate/batch_simulation/annealing/TaN/T300/'
            
        paths,Results = save_simulation(files_copy,dst,n_sim) # Create folders and python files
        
    else:
        paths = {'data': ''}
        Results = []
        
    ovito_file = False

    experiments = ['deposition','annealing']
    experiment = experiments[0]

    if experiment == 'deposition':         
# =============================================================================
#         Experimental conditions
#         
# =============================================================================
        sticking_coeff = 1
        partial_pressure = 0.1 # (Pa = N m^-2 = kg m^-1 s^-2)
        # p = 0.1 - 10 typical values 
        mass_specie = 63.546 # (mass of Copper in u) 
        chemical_specie = 'Cu'
        # T = 573 + n_sim * 100 # (K)
        temp = [300,500,800]
        T = temp[n_sim] # (K)
        
        experimental_conditions = [sticking_coeff,partial_pressure,mass_specie,T,chemical_specie,experiment]
    
# =============================================================================
#         Crystal structure
#         
# =============================================================================
        a = 0.358 # (nm)
        b = 0.358 # (nm)
        c = 0.358 # (nm)
        lattice_constants = (a,b,c)
        crystal_size = (2, 2,1) # (nm)
        bravais_latt = ['fcc']
        orientation = ['001','111']
        lattice_properties = [lattice_constants,crystal_size,bravais_latt[0],orientation[1]]
        
# =============================================================================
#             Superbasin parameters
#     
# =============================================================================
        n_search_superbasin = 3 # If the time step is very small during 10 steps, search for superbasin
        time_step_limits = 1e-8 # Time needed for efficient evolution of the system
        E_min = 0.5
        superbasin_parameters = [n_search_superbasin, time_step_limits,E_min]
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
# 
#       ACTIVATION ENERGIES - Cu
#       Kim, Sung Youb, In-Ho Lee, and Sukky Jun. 
#       "Transition-pathway models of atomic diffusion on fcc metal surfaces. I. Flat surfaces." 
#       Physical Review B 76, no. 24 (2007): 245407.
# 
#       Kim, Sung Youb, In-Ho Lee, and Sukky Jun. 
#       "Transition-pathway models of atomic diffusion on fcc metal surfaces. II. Stepped surfaces." 
#       Physical Review B 76, no. 24 (2007): 245408.
# =============================================================================
        select_dataset = 0   
        Act_E_dataset = ['TaN','Ru25','Ru50','test','homoepitaxial']  
    
        #E_mig_plane_Cu = 0.05*(n_sim+1) # (eV)
        E_dataset = {'TaN':[0.85,0.13,0.13,0.13,0.095,0.19,0.318,0.043,0.477,0.245,0.309],
                  'Ru25':[0.6,0.13,0.13,0.20,0.095,0.23,0.318,0.043,0.477,0.245,0.309],
                  'Ru50':[0.4,0.13,0.13,0.28,0.095,0.38,0.318,0.043,0.477,0.245,0.309],
                   'test':[0.85,0.13,0.13,0.13,0.095,0.2,0.318,0.043,0.477,0.245,0.309],
                   'homoepitaxial':[0.85,0.13,0.13,0.313,0.095,0.528,0.318,0.043,0.477,0.245,0.309]}
        
        E_mig_sub = E_dataset[Act_E_dataset[select_dataset]][0] # (eV)
        E_mig_upward_subs_layer111 = E_dataset[Act_E_dataset[select_dataset]][1]
        E_mig_downward_layer111_subs = E_dataset[Act_E_dataset[select_dataset]][2]
        E_mig_upward_layer1_layer2_111 = E_dataset[Act_E_dataset[select_dataset]][3]
        E_mig_downward_layer2_layer1_111 = E_dataset[Act_E_dataset[select_dataset]][4]
        E_mig_upward_subs_layer100 = E_dataset[Act_E_dataset[select_dataset]][5]
        E_mig_downward_layer100_subs = E_dataset[Act_E_dataset[select_dataset]][6]
        E_mig_111_terrace_Cu = E_dataset[Act_E_dataset[select_dataset]][7]
        E_mig_100_terrace_Cu = E_dataset[Act_E_dataset[select_dataset]][8]
        E_mig_edge_100 = E_dataset[Act_E_dataset[select_dataset]][9]
        E_mig_edge_111 = E_dataset[Act_E_dataset[select_dataset]][10]

             
# =============================================================================
#     Böyükata, M., & Belchior, J. C. (2008). 
#     Structural and Energetic Analysis of Copper Clusters: MD Study of Cu n (n = 2-45). 
#     In J. Braz. Chem. Soc (Vol. 19, Issue 5).
#      - Clustering energy
# 
#     Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2020). 
#     The role of Ru passivation and doping on the barrier and seed layer properties of Ru-modified TaN for copper interconnects. 
#     Journal of Chemical Physics, 152(14). https://doi.org/10.1063/5.0003852
# =============================================================================    
    # clustering_energy = -0.252
    #clustering_energy = -0.21
        clustering_energy = -0.15
        E_clustering = [0,0,clustering_energy * 2,clustering_energy * 3,clustering_energy * 4,clustering_energy * 5,clustering_energy * 6,clustering_energy * 7,clustering_energy * 8,clustering_energy * 9,clustering_energy * 10,clustering_energy * 11,clustering_energy * 12,clustering_energy * 13] 

    

# =============================================================================
#     Papanicolaou, N. 1, & Evangelakis, G. A. (n.d.). 
#     COMPARISON OF DIFFUSION PROCESSES OF Cu AND Au ADA TOMS ON THE Cu(1l1) SURFACE BY MOLECULAR DYNAMICS.
#     
#     Mińkowski, Marcin, and Magdalena A. Załuska-Kotur. 
#     "Diffusion of Cu adatoms and dimers on Cu (111) and Ag (111) surfaces." 
#     Surface Science 642 (2015): 22-32. 10.1016/j.susc.2015.07.026
# =============================================================================

    # Binding energy | Desorption energy: https://doi.org/10.1039/D1SC04708F
    # Surface: [0]-TaN, [1]-Ru25, [2]-Ru50, [3]-Ru100, [4]-1 ML Ru passivation
        binding_energy = {'TaN':0, 'Ru25':-0.05, 'Ru50':-0.15, 'test':-0.00}
        Act_E_list = [E_mig_sub,
                      E_mig_upward_subs_layer111,E_mig_downward_layer111_subs,
                      E_mig_upward_layer1_layer2_111,E_mig_downward_layer2_layer1_111,
                      E_mig_upward_subs_layer100,E_mig_downward_layer100_subs,
                      E_mig_111_terrace_Cu,E_mig_100_terrace_Cu,
                      E_mig_edge_100,E_mig_edge_111,
                      binding_energy[Act_E_dataset[select_dataset]],E_clustering]

# =============================================================================
#     Initialize the crystal grid structure - nodes with empty spaces
# =============================================================================
        Co_latt = Crystal_Lattice(lattice_properties,experimental_conditions,Act_E_list,ovito_file,superbasin_parameters)

        # Maximum probability per site for deposition to establish a timestep limits
        # The maximum timestep is that one that occupy X% of the site during the deposition process
        P_limits = 0.02
        Co_latt.limit_kmc_timestep(P_limits)
    
# =============================================================================
#     - test[0] - Normal deposition
#     - test[1] - Introduce a single particle in a determined site
#     - test[2] - Introduce and remove a single particle in a determined site 
#     - test[3] - Introduce two adjacent particles
#     - test[4] - Hexagonal seed - 7 particles in plane + 1 particle in plane
#     - test[5] - Hexagonal seed - 7 particles in plane and 1 on the top of the layer
#     - test[6] - 2 hexagonal seeds - 2 layers and one particle on the top 
#     - test[7] - 2 hexagonal seeds - 2 layers and one particle attach to the lateral
#     - test[8] - cluster
# =============================================================================
        test_selected = 0
        test = [0,1,2,3,4,5,6,7,8]

        # Deposition process of chemical species
        if Co_latt.timestep_limits < float('inf'):
            Co_latt.deposition_specie(Co_latt.timestep_limits,rng,test[test_selected])
            Co_latt.track_time(Co_latt.timestep_limits) 
            Co_latt.add_time()
        else:
            Co_latt.deposition_specie(0,rng,test[test_selected])
            Co_latt.track_time(0) 
            Co_latt.add_time()
            
    elif experiment == 'annealing':
        
        import pickle
        path = r'/sfihome/samuel.delgado/Copper_deposition/Varying_substrate/annealing/TaN/T500/'
        filename = path + 'variables.pkl'
        
        # Open the file in binary mode
        with open(filename, 'rb') as file:
          
            # Call load method to deserialze
            myvar = pickle.load(file)
            
        Co_latt = myvar['Co_latt']
        
        temp = [300,500,800] #(K)
        
        Co_latt.temperature = temp[n_sim]
        Co_latt.experiment = experiment
        P_limits = 1
        Co_latt.limit_kmc_timestep(P_limits)
        Co_latt.time = 0
        Co_latt.list_time = []

    return Co_latt,rng,paths,Results


def search_superbasin(Co_latt):
          
    # We need a deepcopy? Co_latt.sites_occupied will be modified on site
    # when calling Superbasin() and it will change the order of sites_occupied
    # sites_occupied = copy.deepcopy(Co_latt.sites_occupied)
    
    # This approach should be more efficient and memory-friendly
    sites_occupied = Co_latt.sites_occupied[:] 

    for idx in sites_occupied:
        for event in Co_latt.grid_crystal[idx].site_events:
            if (idx not in Co_latt.superbasin_dict) and (event[3] <= Co_latt.E_min):
                Co_latt.superbasin_dict.update({idx: Superbasin(idx, Co_latt, Co_latt.E_min)})
                

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
            f.write('Time,Mass,Sites Occupation,Average Thickness,Terrace Area,std_terrace,max_terrace,RMS Roughness,Performance time\n')
    
    def measurements_crystal(self, time, mass_gained, sites_occupation, thickness, avg_terrace,std_terrace,max_terrace, surf_roughness_RMS,performance_time):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{time},{mass_gained},{sites_occupation},{thickness},{avg_terrace},{std_terrace},{max_terrace},{surf_roughness_RMS},{performance_time}\n')