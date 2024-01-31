# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""
from initialization import initialization,save_variables
from KMC import KMC

save_data =  False

n_sim = 1

Co_latt,rng,paths = initialization(n_sim,save_data)

print(Co_latt.time)
Co_latt.add_time()
Co_latt.plot_crystal(45,45)

j = 0
for i in range(2):
    Co_latt = KMC(Co_latt,rng)
    
    if i%1 == 0:
        j+=1
        Co_latt.plot_crystal(45,45,paths['data'],j)      
        Co_latt.add_time()
        print('Total time: ',Co_latt.list_time[-1])
        Co_latt.deposition_specie(Co_latt.list_time[-1] - Co_latt.list_time[-2],rng)

    
Co_latt.plot_crystal(45,45)

# Variables to save
variables = {'Co_latt' : Co_latt}
if save_data: save_variables(paths['program'],variables)
    


