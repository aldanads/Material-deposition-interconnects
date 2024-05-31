# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""
from initialization import initialization,save_variables
from KMC import KMC
import numpy as np
import time

save_data = True

for n_sim in range(0,3):
    

    Co_latt,rng,paths,Results = initialization(n_sim,save_data)
    
    Co_latt.add_time()
    Co_latt.plot_crystal(45,45)
    

    j = 0
    n_part = len(Co_latt.sites_occupied)
    nothing_happen = 0
    # total_steps = int(1e5)
    # snapshoots_steps = int(1e3)
    total_steps = int(5e5)
    time_limit = 0.01 # s
    thickness_limit = 1 # (1 nm)
    Co_latt.measurements_crystal()
    i = 0
    snapshoots_steps = int(5e3)

    starting_time = time.time()

    #while Co_latt.list_time[-1] < time_limit:
    while Co_latt.thickness < thickness_limit:
        i+=1
    #for i in range(total_steps):

        Co_latt,KMC_time_step = KMC(Co_latt,rng)
        Co_latt.deposition_specie(KMC_time_step,rng)
    
        if i%snapshoots_steps== 0:

            # If there is only migration for many kMC steps, we increase once the timestep 
            # for the deposition 
            if len(Co_latt.sites_occupied) == n_part:
                nothing_happen +=1
                if nothing_happen == 4:
                    Co_latt.deposition_specie(Co_latt.timestep_limits,rng)
                    if Co_latt.timestep_limits < float('Inf'):
                        Co_latt.track_time(Co_latt.timestep_limits)
                        Co_latt.add_time()
                    else:
                        Co_latt.add_time()

            else: 
                n_part = len(Co_latt.sites_occupied)
                nothing_happen = 0
                Co_latt.add_time()
            
            j+=1
            Co_latt.measurements_crystal()
            # print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',Co_latt.list_time[-1])
            # print(str(Co_latt.list_time[-1]/time_limit * 100) + ' %','| Total time: ',Co_latt.list_time[-1])
            print(str(Co_latt.thickness/thickness_limit * 100) + ' %','| Thickness: ', Co_latt.thickness, '| Total time: ',Co_latt.list_time[-1])
            end_time = time.time()
            if save_data:
                Results.measurements_crystal(Co_latt.list_time[-1],Co_latt.mass_gained,Co_latt.fraction_sites_occupied,
                                              Co_latt.thickness,np.mean(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) != 0]),Co_latt.surf_roughness_RMS,end_time-starting_time)

            Co_latt.plot_crystal(45,45,paths['data'],j) 

    Co_latt.plot_crystal(45,45)
    
    # Variables to save
    variables = {'Co_latt' : Co_latt}
    if save_data: save_variables(paths['program'],variables)
    

