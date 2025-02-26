# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:43:50 2024

@author: samuel.delgado
"""
from balanced_tree import Node, build_tree, update_data, search_value
import numpy as np

def KMC(System_state,rng):
        
    time_step = 0
    grid_crystal = System_state.grid_crystal
    superbasin_dict = System_state.superbasin_dict

# =============================================================================
#     TR_catalog store:
#      - TR_catalog[0] = TR
#      - TR_catalog[1] = Arrival site
#      - TR_catalog[2] = Event label - Migration, desorption, etc
#      - TR_catalog[3] = Starting site
# =============================================================================
    TR_catalog = []

    for idx in System_state.sites_occupied + System_state.adsorption_sites:
        
        if idx not in superbasin_dict:
            TR_catalog.extend([(item[0],item[1],item[2],idx) for item in grid_crystal[idx].site_events])
        else:
            TR_catalog.extend([(item[0],item[1],item[2],idx) for item in superbasin_dict[idx].site_events_absorbing])


    # Sort the list of events
    sorted(TR_catalog,key = lambda x:x[0])
    
    
    # Build a balanced tree structure
    TR_tree = build_tree(TR_catalog)
    # Each node is the sum of their children, starting from the leaf
    sumTR = update_data(TR_tree)
    

    if sumTR == None: return System_state,time_step # Exit if there is not possible event
    # When we only have one node in the tree, it returns a tuple
    if type(sumTR) is tuple: sumTR = sumTR[0]
    # We search in our binary tree the event that happen

    chosen_event = search_value(TR_tree,sumTR*rng.random())

    #Calculate the time step
    time_step += -np.log(rng.random())/sumTR
    # If the time step is big because of the TR, we need to allow the deposition process to occur
    # We establish a time step limits that the deposition is relevant
    if time_step > System_state.timestep_limits:
        time_step = System_state.timestep_limits
        if rng.random() < 1-np.exp(-sumTR*time_step):
            System_state.processes(chosen_event)

    else:
        System_state.processes(chosen_event)
        
    System_state.track_time(time_step)  
    System_state.update_superbasin(chosen_event)
    

    return System_state,time_step