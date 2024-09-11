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
            f.write('Substrate,Partial_pressure,Temperature,Thickness,Ra_roughness,z_mean,RMS_roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace\n')
    
    def measurements_crystal(self, Substrate, folder_P,temperature,thickness,Ra_roughness,z_mean,roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{Substrate},{folder_P},{temperature},{thickness},{Ra_roughness},{z_mean},{roughness},{n_peaks},{mean_size_peak},{std_size_peak},{max_size_peak},{mean_island_terraces},{std_island_terraces},{max_island_terrace},{general_terrace},{std_terrace},{max_terrace}\n')




# Path to the variables files, for every folder and Sim_i
path = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Batch simulations\Thickness limit\Time_evolution\\'
#folder_P = '\P=0.1'
folder_subs = os.listdir(path)
path_2 = r'\Program\\'

# Data saved in a OS
system = ['Windows','Linux']
choose_system = system[1]

# Store the data of each file in a list
dfs = []
temperature = {'Sim_0':300, 'Sim_1':500, 'Sim_2':800}

name_generated_file = 'Figure_postAnn.csv'
Results = SimulationResults(name_generated_file)

name_histogram_size_file_terraces = 'Histogram_terraces_postAnn.csv'
dfs_histogram_terraces = []

name_histogram_size_file_neighbors = 'Histogram_neighbors_postAnn.csv'
dfs_histogram_neighbors = []

occ_rate_filename = 'Occupation_rate_postAnn.csv'
dfs_occ_rate = []

for subs in folder_subs:
    
    for folder_P in os.listdir(path+subs):
    
        folder_1 = os.listdir(path+subs + r'\\' + folder_P)
        
        for folder in folder_1:
    
            if choose_system == 'Windows':
                import shelve
                
                filename = path + subs + r'\\' + folder_P + r'\\' + folder + path_2 + 'variables2'
    
                my_shelf = shelve.open(filename)
                for key in my_shelf:
                    globals()[key]=my_shelf[key]
                my_shelf.close()
                
            elif choose_system == 'Linux':
                
                import pickle
                filename = path + subs + r'\\' + folder_P + r'\\' + folder + path_2 + 'variables.pkl'
                
                # Open the file in binary mode
                with open(filename, 'rb') as file:
                    print(filename)
                    # Call load method to deserialze
                    myvar = pickle.load(file)
                    
                Co_latt = myvar['Co_latt']
                Co_latt.peak_detection()
                Co_latt.islands_analysis()
                Co_latt.neighbors_calculation()
                Co_latt.RMS_roughness()

                peak_size = []
                for peak in Co_latt.peak_list:
                    if len(peak.island_sites) > 10:
                        peak_size.append(len(peak.island_sites))
                                        
                peak_mean_size = np.mean(peak_size)
                peak_std_size = np.std(peak_size)
                islands_terraces = []
                Co_latt.terrace_area()
                for island in Co_latt.islands_list:
                    island.layers_calculation(Co_latt)
                    islands_terraces.append(np.mean(np.array(island.terraces)[np.array(island.terraces) != 0]))
                
                # Results: thickness, roughness, islands, terraces
                peak_size_max = max(peak_size) if peak_size else 0
                Results.measurements_crystal(subs,folder_P,temperature[folder],Co_latt.thickness,Co_latt.Ra_roughness,Co_latt.z_mean,Co_latt.surf_roughness_RMS,
                                             len(peak_size),np.mean(peak_size),np.std(peak_size),peak_size_max
                                             ,np.mean(islands_terraces),np.std(islands_terraces),max(islands_terraces),np.mean(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),np.std(np.array(Co_latt.terraces)[np.array(Co_latt.terraces) > 0]),max(Co_latt.terraces))
                
                # Size of terrace per layer
                df_histogram_terraces = pd.DataFrame({subs + '_' + folder_P + '_' + str(temperature[folder]) + '_terrace_area': Co_latt.terraces})   
                dfs_histogram_terraces.append(df_histogram_terraces)
                
                # Histogram of neighbors
                df_histogram_neighbors = pd.DataFrame({subs + '_' + folder_P + '_' + str(temperature[folder]) + '_neighbors': Co_latt.histogram_neighbors})   
                dfs_histogram_neighbors.append(df_histogram_neighbors)
                
                # Ocuppation rate per layer
                df_occ_rate = pd.DataFrame({subs + '_' + folder_P + '_' + str(temperature[folder]) + '_occupation_rate': Co_latt.layers[1]})   
                dfs_occ_rate.append(df_occ_rate)
                

df_combined = pd.concat(dfs_histogram_terraces, axis=1)    
df_combined.to_csv(name_histogram_size_file_terraces, index=False)

df_combined = pd.concat( dfs_histogram_neighbors, axis=1)    
df_combined.to_csv(name_histogram_size_file_neighbors, index=False)

df_combined = pd.concat(dfs_occ_rate, axis=1)    
df_combined.to_csv(occ_rate_filename, index=False)
    
