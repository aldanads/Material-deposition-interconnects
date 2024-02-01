# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:43:50 2024

@author: samuel.delgado
"""
from balanced_tree import Node, build_tree, update_data, search_value
import numpy as np

def KMC(Co_latt,rng):
        
    time = 0
    grid_crystal = Co_latt.grid_crystal
# =============================================================================
#     TR_catalog store:
#      - TR_catalog[0] = TR
#      - TR_catalog[1] = Arrival site
#      - TR_catalog[2] = Event label - Migration, desorption, etc
#      - TR_catalog[3] = Starting site
# =============================================================================
    TR_catalog = []

    for idx in Co_latt.sites_occupied:
        TR_catalog.extend([(item[0],item[1],item[2],idx) for item in grid_crystal[idx].site_events])
    
    # Sort the list of events
    sorted(TR_catalog,key = lambda x:x[0])
    # Build a balanced tree structure
    TR_tree = build_tree(TR_catalog)
    # Each node is the sum of their children, starting from the leaf
    sumTR = update_data(TR_tree)
    

    if sumTR == None: return Co_latt,time # Exit if there is not possible event
    # When we only have one node in the tree, it returns a tuple
    if type(sumTR) is tuple: sumTR = sumTR[0]
    # We search in our binary tree the events that happen
    chosen_event = search_value(TR_tree,sumTR*rng.random())
    #Calculate the time step
    time += -np.log(rng.random())/sumTR

    print(time,Co_latt.timestep_limits)
    if time > Co_latt.timestep_limits:
        time = Co_latt.timestep_limits
        if rng.random() < 1-np.exp(-sumTR*time):
            Co_latt.processes(chosen_event)
            """Can be inside Co_latt????"""
    else:
        Co_latt.processes(chosen_event)
    Co_latt.track_time(time)  
   
    return Co_latt,time