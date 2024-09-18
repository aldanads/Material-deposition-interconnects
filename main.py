# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
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
        thickness_limit = 1 # (1 nm)
        Co_latt.measurements_crystal()
        i = 0

        while Co_latt.thickness < thickness_limit:
            i+=1
            Co_latt,KMC_time_step = KMC(Co_latt,rng)
            list_time_step.append(KMC_time_step)
            print(KMC_time_step/Co_latt.timestep_limits)
            Co_latt.deposition_specie(KMC_time_step,rng)
            if np.mean(list_time_step[-Co_latt.n_search_superbasin:]) <= Co_latt.time_step_limits:
                nothing_happen +=1    
            else:
                nothing_happen = 0
            
            if nothing_happen >= Co_latt.n_search_superbasin:
                for idx in Co_latt.sites_occupied:
                    for event in Co_latt.grid_crystal[idx].site_events:
                        
                        if (idx not in Co_latt.superbasin_dict) and (event[3] <= Co_latt.E_min):
                            print(idx)
                print(Co_latt.superbasin_dict.keys())
                
                search_superbasin(Co_latt)
                nothing_happen = 0
                
                for idx in Co_latt.sites_occupied:
                    for event in Co_latt.grid_crystal[idx].site_events:
                        
                        if (idx not in Co_latt.superbasin_dict) and (event[3] <= Co_latt.E_min):
                            print(idx)
                            
                print(Co_latt.superbasin_dict.keys())
                    # sys.exit()
        
            if i%snapshoots_steps== 0:
    
                # If there is only migration for many kMC steps, we increase once the timestep 
                # for the deposition 
                # if len(Co_latt.sites_occupied) == n_part:
                #     nothing_happen +=1
                #     if nothing_happen == 4:
                #         Co_latt.deposition_specie(Co_latt.timestep_limits,rng)
                #         if Co_latt.timestep_limits < float('Inf'):
                #             Co_latt.track_time(Co_latt.timestep_limits)
                #             Co_latt.add_time()
                #         else:
                #             Co_latt.add_time()
    
                # else: 
                #     n_part = len(Co_latt.sites_occupied)
                #     nothing_happen = 0
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


# # Use cProfile to profile the main function
# if __name__ == '__main__':
#     cProfile.run('main()', 'profile_output.prof')    

# import pstats

# # Load and analyze the profiling results
# p = pstats.Stats('profile_output.prof')
# p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions