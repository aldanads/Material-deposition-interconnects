# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""


"""
Profiling - Reference: Grown 0.4 nm in a 2x2nm box

         2676429707 function calls (2668330686 primitive calls) in 9996.511 seconds

   Ordered by: internal time
   List reduced from 12819 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 87014705 1697.435    0.000 1993.252    0.000 crystal_lattice.py:321(available_adsorption_sites)
 98945669  966.934    0.000 1526.271    0.000 Site.py:160(available_migrations)
 87014705  785.962    0.000 1871.621    0.000 Site.py:331(detect_planes)
 98945669  760.480    0.000  915.542    0.000 Site.py:418(transition_rates)
 87014705  746.771    0.000 3516.502    0.000 Site.py:95(supported_by)
 87014705  527.334    0.000  700.369    0.000 Site.py:380(detect_edges)
-1821479170/-1821489333  440.519   -0.000  440.525   -0.000 {built-in method builtins.len}
  4458920  400.697    0.000 8351.543    0.002 crystal_lattice.py:658(update_sites)
326221503  400.266    0.000  400.266    0.000 Site.py:336(<lambda>)
 88581059  367.008    0.000  767.598    0.000 {built-in method builtins.sorted}
397918711  297.828    0.000  375.677    0.000 Site.py:115(calculate_clustering_energy)
 93885924  279.128    0.000  279.166    0.000 {built-in method numpy.array}
    13224  215.039    0.016  474.352    0.036 superbasin.py:107(transition_matrix)
619315526  213.524    0.000  213.545    0.000 {built-in method builtins.max}
  4458667  180.502    0.000  284.071    0.000 crystal_lattice.py:748(remove_specie_site)

         9034848122 function calls (9028225809 primitive calls) in 13273.455 seconds

   Ordered by: internal time
   List reduced from 13082 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 90864365 1873.830    0.000 2201.324    0.000 crystal_lattice.py:321(available_adsorption_sites)
1502205239 1433.478    0.000 1433.478    0.000 Site.py:143(<lambda>)
591363342 1429.986    0.000 4026.251    0.000 {built-in method builtins.sorted}
 99657161 1097.536    0.000 4026.995    0.000 Site.py:183(available_migrations)
 90864365  817.853    0.000 4350.333    0.000 Site.py:96(supported_by)
 90864365  810.481    0.000 1978.978    0.000 Site.py:354(detect_planes)
407948350  668.110    0.000 3291.556    0.000 Site.py:116(calculate_clustering_energy)
333618222  427.285    0.000  427.285    0.000 Site.py:359(<lambda>)
  4642156  426.719    0.000 11562.122    0.002 crystal_lattice.py:658(update_sites)
2124822805/2124814785  413.282    0.000  413.287    0.000 {built-in method builtins.len}
353878868  390.040    0.000  390.040    0.000 Site.py:405(<lambda>)
 99657161  383.317    0.000  557.904    0.000 Site.py:447(transition_rates)
353878868  345.214    0.000  345.214    0.000 Site.py:125(<lambda>)
 97655938  319.262    0.000  319.284    0.000 {built-in method numpy.array}
    13463  238.244    0.018  520.361    0.039 superbasin.py:107(transition_matrix)

"""


import cProfile
import sys

def main():
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
        
        snapshoots_steps = int(2e1)
        starting_time = time.time()
    # =============================================================================
    #     Deposition
    # 
    # =============================================================================
        if System_state.experiment == 'deposition':   
            
            nothing_happen = 0
            list_time_step = []
            thickness_limit = 4 # (1 nm)
            System_state.measurements_crystal()
            i = 0
            while System_state.thickness < thickness_limit:
                i+=1
                System_state,KMC_time_step = KMC(System_state,rng)
                # quit()
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
if __name__ == '__main__':
    cProfile.run('main()', 'profile_output.prof')    

import pstats

# # Load and analyze the profiling results
p = pstats.Stats('profile_output.prof')
p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions