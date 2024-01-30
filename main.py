# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""
from initialization import initialization
from KMC import KMC

save_data =  False

n_sim = 1

Co_latt,rng = initialization(n_sim,save_data)

print(Co_latt.time)
Co_latt.add_time()
Co_latt.plot_crystal()

for i in range(100000):
    Co_latt = KMC(Co_latt,rng)
    
    if i%1000 == 0:
        Co_latt.plot_crystal(60,10)      
        Co_latt.add_time()
        print('Time between depositions: ',Co_latt.list_time[-1] - Co_latt.list_time[-2])
        Co_latt.deposition_specie(Co_latt.list_time[-1] - Co_latt.list_time[-2],rng)

    
Co_latt.deposition_specie(Co_latt.time,rng)
print(Co_latt.time)
Co_latt.plot_crystal()
    


