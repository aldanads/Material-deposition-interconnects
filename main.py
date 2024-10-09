# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""

"""
4719396067 function calls (4710390063 primitive calls) in 26906.461 seconds

   Ordered by: internal time
   List reduced from 2010 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 55205739 8618.828    0.000 8754.190    0.000 Site.py:355(detect_edges)
 55205739 3611.556    0.000 3736.298    0.000 crystal_lattice.py:202(available_adsorption_sites)
 61491592 3262.338    0.000 3816.401    0.000 Site.py:156(available_migrations)
 61491592 2269.977    0.000 3489.035    0.000 Site.py:414(transition_rates)
191305668 1501.699    0.000 1501.699    0.000 Site.py:313(<lambda>)
 61491592 1131.087    0.000 1131.087    0.000 Site.py:419(<listcomp>)
 55205739 1070.504    0.000 13034.189    0.000 Site.py:91(supported_by)
 55205739  708.300    0.000 3045.069    0.000 Site.py:308(detect_planes)
    13175  616.443    0.047  617.056    0.047 linalg.py:1499(svd)
273373586  449.113    0.000  489.033    0.000 Site.py:111(calculate_clustering_energy)
  2887455  406.297    0.000 24482.225    0.008 crystal_lattice.py:488(update_sites)
 56335730  345.652    0.000 1847.905    0.000 {built-in method builtins.sorted}
 56476355  267.114    0.000  267.193    0.000 {built-in method numpy.array}
1568718429/1568714227  213.142    0.000  213.147    0.000 {built-in method builtins.len}
 94901339  195.571    0.000  343.297    0.000 {method 'update' of 'set' objects}

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


import cProfile
import sys

def main():
    from initialization import initialization,save_variables,search_superbasin
    from KMC import KMC
    import numpy as np
    import time
    
    save_data = False
    
    for n_sim in range(0,1):
        
    
        Co_latt,rng,paths,Results = initialization(n_sim,save_data)
        Co_latt.add_time()
    
        Co_latt.plot_crystal(45,45)    
        j = 0
        
        snapshoots_steps = int(1e1)
        starting_time = time.time()
    
    # =============================================================================
    #     Deposition
    # 
    # =============================================================================
        if Co_latt.experiment == 'deposition':   
            
            nothing_happen = 0
            list_time_step = []
            thickness_limit = 0.3 # (1 nm)
            Co_latt.measurements_crystal()
            i = 0
    
            while Co_latt.thickness < thickness_limit:
                i+=1
                Co_latt,KMC_time_step = KMC(Co_latt,rng)
                list_time_step.append(KMC_time_step)
                Co_latt.deposition_specie(KMC_time_step,rng)
                if np.mean(list_time_step[-Co_latt.n_search_superbasin:]) <= Co_latt.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                
                if nothing_happen >= Co_latt.n_search_superbasin:
                    search_superbasin(Co_latt)
                    nothing_happen = 0
    
            
                if i%snapshoots_steps== 0:
                    Co_latt.add_time()
                    
                    j+=1
                    Co_latt.measurements_crystal()
                    # print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',Co_latt.list_time[-1])
                    # print(str(Co_latt.list_time[-1]/time_limit * 100) + ' %','| Total time: ',Co_latt.list_time[-1])
                    print(str(Co_latt.thickness/thickness_limit * 100) + ' %','| Thickness: ', Co_latt.thickness, '| Total time: ',Co_latt.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(Co_latt.list_time[-1],Co_latt.mass_gained,Co_latt.fraction_sites_occupied,
                                                      Co_latt.thickness,np.mean(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),np.std(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),max(Co_latt.terraces),
                                                      Co_latt.surf_roughness_RMS,end_time-starting_time)
        
                    Co_latt.plot_crystal(45,45,paths['data'],j)
                    
                    # print('j = ',j)
                    # if j >= 20:
                    #     sys.exit()
    
    # =============================================================================
    #     Annealing  
    #            
    # =============================================================================
        elif Co_latt.experiment == 'annealing':
            i = 0
            #otal_steps = int(2.5e6)
            total_steps = int(100)
            Co_latt.measurements_crystal()
            
            while j*snapshoots_steps < total_steps:
    
                i+=1
                Co_latt,KMC_time_step = KMC(Co_latt,rng)
                
                if i%snapshoots_steps== 0:
                    Co_latt.add_time()
                    j+=1
                    Co_latt.measurements_crystal()
                    print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',Co_latt.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(Co_latt.list_time[-1],Co_latt.mass_gained,Co_latt.fraction_sites_occupied,
                                                      Co_latt.thickness,np.mean(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),np.std(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),max(Co_latt.terraces),
                                                      Co_latt.surf_roughness_RMS,end_time-starting_time)
                        
                    Co_latt.plot_crystal(45,45,paths['data'],j)
    
    
        Co_latt.plot_crystal(45,45)
        
        # Variables to save
        variables = {'Co_latt' : Co_latt}
        if save_data: save_variables(paths['program'],variables)


# Use cProfile to profile the main function
if __name__ == '__main__':
    cProfile.run('main()', 'profile_output.prof')    

import pstats

# Load and analyze the profiling results
p = pstats.Stats('profile_output.prof')
p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions