# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""


"""
Profiling - Reference: Grown 0.3 nm in a 2x2nm box

1606142554 function calls (1601344011 primitive calls) in 3808.223 seconds

   Ordered by: internal time
   List reduced from 1997 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 21182531  727.770    0.000  773.141    0.000 crystal_lattice.py:210(available_adsorption_sites)
 20454173  439.688    0.000  593.239    0.000 Site.py:155(available_migrations)
 21182531  363.121    0.000 1312.305    0.000 Site.py:90(supported_by)
 21182531  229.669    0.000  632.346    0.000 Site.py:318(detect_planes)
 21182531  213.482    0.000  260.788    0.000 Site.py:365(detect_edges)
 20454173  148.285    0.000  148.285    0.000 Site.py:429(<listcomp>)
 88225131  130.736    0.000  142.564    0.000 Site.py:110(calculate_clustering_energy)
  1125501  126.356    0.000 3090.793    0.003 crystal_lattice.py:511(update_sites)
 67895751  116.200    0.000  116.200    0.000 Site.py:323(<lambda>)
 21732308  114.275    0.000  230.751    0.000 {built-in method builtins.sorted}
 20454173  113.631    0.000  285.750    0.000 Site.py:424(transition_rates)
     2550   93.557    0.037   93.557    0.037 {method 'encode' of 'ImagingEncoder' objects}
 22067444   85.175    0.000   85.215    0.000 {built-in method numpy.array}
     5819   65.861    0.011  121.798    0.021 superbasin.py:107(transition_matrix)
535043268/535040492   64.367    0.000   64.370    0.000 {built-in method builtins.len}
"""

"""
Cache for:
    - calculate_clustering_energy
    - transition rates
    - detect_edges
    
Optimize:
    - available_adsorption_sites --> cache?
    - available_migration --> cache?
"""

"""
TASKS now:
    - Problem with neighbors --> Some of them are not created
"""
"""
REFACTOR CRYSTAL EDGE AND CRYSTALLOGRAPHIC PLANES IN GENERAL
After using pymatgen, all those things changed
"""


import cProfile
import sys

# def main():
from initialization import initialization,save_variables,search_superbasin
from KMC import KMC
import numpy as np
import time

save_data = False

for n_sim in range(0,1):
    

    System_state,rng,paths,Results = initialization(n_sim,save_data)
    System_state.add_time()

    System_state.plot_crystal(45,45)    
    j = 0
    
    snapshoots_steps = int(1e1)
    starting_time = time.time()
# =============================================================================
#     Deposition
# 
# =============================================================================
    if System_state.experiment == 'deposition':   
        
        nothing_happen = 0
        list_time_step = []
        thickness_limit = 30 # (1 nm)
        System_state.measurements_crystal()
        i = 0
        while System_state.thickness < thickness_limit:
            i+=1
            System_state,KMC_time_step = KMC(System_state,rng)
            quit()
            list_time_step.append(KMC_time_step)
            if np.mean(list_time_step[-System_state.n_search_superbasin:]) <= System_state.time_step_limits:
                nothing_happen +=1    
            else:
                nothing_happen = 0
            
            if nothing_happen >= System_state.n_search_superbasin:
                search_superbasin(System_state)
                nothing_happen = 0

        
            if i%snapshoots_steps== 0:
                System_state.add_time()
                
                j+=1
                System_state.measurements_crystal()
                # print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                # print(str(System_state.list_time[-1]/time_limit * 100) + ' %','| Total time: ',System_state.list_time[-1])
                print(str(System_state.thickness/thickness_limit * 100) + ' %','| Thickness: ', System_state.thickness, '| Total time: ',System_state.list_time[-1])
                end_time = time.time()
                if save_data:
                    Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                  System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                  System_state.surf_roughness_RMS,end_time-starting_time)
    
                System_state.plot_crystal(45,45,paths['data'],j)
                
                # print('j = ',j)
                # if j == 2:
                #     sys.exit()

# =============================================================================
#     Annealing  
#            
# =============================================================================
    elif System_state.experiment == 'annealing':
        i = 0
        #otal_steps = int(2.5e6)
        total_steps = int(100)
        System_state.measurements_crystal()
        
        while j*snapshoots_steps < total_steps:

            i+=1
            System_state,KMC_time_step = KMC(System_state,rng)
            
            if i%snapshoots_steps== 0:
                System_state.add_time()
                j+=1
                System_state.measurements_crystal()
                print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                end_time = time.time()
                if save_data:
                    Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                  System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                  System_state.surf_roughness_RMS,end_time-starting_time)
                    
                System_state.plot_crystal(45,45,paths['data'],j)


    System_state.plot_crystal(45,45)
    
    # Variables to save
    variables = {'System_state' : System_state}
    if save_data: save_variables(paths['program'],variables)


# # Use cProfile to profile the main function
# if __name__ == '__main__':
#     cProfile.run('main()', 'profile_output.prof')    

# import pstats

# # Load and analyze the profiling results
# p = pstats.Stats('profile_output.prof')
# p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions