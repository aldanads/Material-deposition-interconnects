# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:35:07 2024

@author: samuel.delgado
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


class SimulationResults:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        # Initialize a CSV file with headers
        with open(excel_filename, 'w') as f:
            f.write('Substrate,Partial_pressure,Temperature,Thickness,RMS_roughness,n_islands,mean_size_islands,std_size_islands,max_size_island,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace\n')
    
    def measurements_crystal(self, Substrate, folder_P,temperature,thickness,roughness,n_islands,mean_size_islands,std_size_islands,max_size_island,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{Substrate},{folder_P},{temperature},{thickness},{roughness},{n_islands},{mean_size_islands},{std_size_islands},{max_size_island},{mean_island_terraces},{std_island_terraces},{max_island_terrace},{general_terrace}\n')




# Path to the variables files, for every folder and Sim_i
path = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Batch simulations\Thickness limit\\'
folder_P = '\P=0.1'
folder_subs = os.listdir(path)
path_2 = r'\Program\\'

# Data saved in a OS
system = ['Windows','Linux']
choose_system = system[1]

# Store the data of each file in a list
dfs = []
temperature = {'Sim_0':300, 'Sim_1':500, 'Sim_2':800}

name_generated_file = 'Figure.csv'
Results = SimulationResults(name_generated_file)

name_histogram_size_file = 'Histogram.csv'
dfs_histogram = []

occ_rate_filename = 'Occupation_rate.csv'
dfs_occ_rate = []

for subs in folder_subs:
    
    folder_1 = os.listdir(path+subs+folder_P)
    
    for folder in folder_1:

        if choose_system == 'Windows':
            import shelve
            
            filename = path + subs+ folder_P + r'\\' + folder + path_2 + 'variables'

            my_shelf = shelve.open(filename)
            for key in my_shelf:
                globals()[key]=my_shelf[key]
            my_shelf.close()
            
        elif choose_system == 'Linux':
            
            import pickle
            filename = path + subs+ folder_P + r'\\' + folder + path_2 + 'variables.pkl'
            print(filename)
            
            # Open the file in binary mode
            with open(filename, 'rb') as file:
              
                # Call load method to deserialze
                myvar = pickle.load(file)
                
            Co_latt = myvar['Co_latt']
            
            Co_latt.islands_analysis()
            island_size = []
            for island in Co_latt.islands_list:
                island_size.append(len(island.island_sites))
                
            island_mean_size = np.mean(island_size)
            island_std_size = np.std(island_size)
            
            islands_terraces = []
            for island in Co_latt.islands_list:
                island.layers_calculation(Co_latt)
                islands_terraces.append(np.mean(np.array(island.terraces)[np.array(island.terraces) != 0]))
            
            # Results: thickness, roughness, islands, terraces
            Results.measurements_crystal(subs,folder_P[1:],temperature[folder],Co_latt.thickness,Co_latt.surf_roughness_RMS,
                                         len(Co_latt.islands_list),np.mean(island_size),np.std(island_size),max(island_size)
                                         ,np.mean(islands_terraces),np.std(islands_terraces),max(islands_terraces),np.mean(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) != 0]))
            
            # Size of islands in a list
            df_histogram = pd.DataFrame({subs + '_' + str(temperature[folder]) + '_island_size': island_size})   
            dfs_histogram.append(df_histogram)
            
            # Ocuppation rate per layer
            df_occ_rate = pd.DataFrame({subs + '_' + str(temperature[folder]) + '_occupation_rate': Co_latt.layers[1]})   
            dfs_occ_rate.append(df_occ_rate)

df_combined = pd.concat(dfs_histogram, axis=1)    
df_combined.to_csv(name_histogram_size_file, index=False)

df_combined = pd.concat(dfs_occ_rate, axis=1)    
df_combined.to_csv(occ_rate_filename, index=False)
    
