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

for n_sim in range(8):
    

    Co_latt,rng,paths,Results = initialization(n_sim,save_data)
    
    print(Co_latt.time)
    Co_latt.add_time()
    Co_latt.plot_crystal(45,45)
    
    j = 0
    n_part = len(Co_latt.sites_occupied)
    nothing_happen = 0
    total_steps = int(5e5)
    snapshoots_steps = int(5e3)

    starting_time = time.time()

    for i in range(total_steps):

        Co_latt,KMC_time_step = KMC(Co_latt,rng)
        Co_latt.deposition_specie(KMC_time_step,rng)
    
        if i%snapshoots_steps== 0:
            j+=1

            Co_latt.plot_crystal(45,45,paths['data'],j) 

            Co_latt.add_time()
            Co_latt.measurements_crystal()
            print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',Co_latt.list_time[-1])
            end_time = time.time()
            Results.measurements_crystal(Co_latt.list_time[-1],Co_latt.mass_gained,Co_latt.fraction_sites_occupied,
                                          Co_latt.thickness,np.mean(Co_latt.terraces),Co_latt.surf_roughness_RMS,end_time-starting_time)
            
            # If there is only migration for many kMC steps, we increase once the timestep 
            # for the deposition 
            if len(Co_latt.sites_occupied) == n_part:
                nothing_happen +=1
                if nothing_happen == 4:
                    Co_latt.deposition_specie(Co_latt.timestep_limits,rng)
            else: 
                n_part = len(Co_latt.sites_occupied)
                nothing_happen = 0
                    
        
    Co_latt.plot_crystal(45,45)
    
    # Variables to save
    variables = {'Co_latt' : Co_latt}
    if save_data: save_variables(paths['program'],variables)
    


