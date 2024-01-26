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
Co_latt.plot_crystal()
for i in range(50):
    Co_latt = KMC(Co_latt,rng)
    
    if i%1 == 0:
        Co_latt.plot_crystal()
        """
        I am using the total time for the "deposition_specie()" and it should be the 
        KMC time step
        """
        Co_latt.deposition_specie(Co_latt.time,rng)
        print(len(Co_latt.sites_occupied) - len(set(Co_latt.sites_occupied)))

    
Co_latt.deposition_specie(Co_latt.time,rng)
print(Co_latt.time)
Co_latt.plot_crystal()
    


