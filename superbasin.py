# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:31:56 2024

@author: samuel.delgado
"""
import numpy as np
from scipy import constants


# =============================================================================
# Fichthorn, Kristen A., and Yangzheng Lin. 
# "A local superbasin kinetic Monte Carlo method." 
# The Journal of chemical physics 138, no. 16 (2013).

# Novotny, M. A. 
# "A tutorial on advanced dynamic Monte Carlo methods for systems with discrete state spaces." 
# Annual Reviews Of Computational PhysicsIX (2001): 153-210.
# =============================================================================

class Superbasin():
    
    def __init__(self,idx, System_state,E_min,sites_occupied):
        
        self.particle_idx = idx
        self.E_min = E_min
        # Make a deep copy of System_state to avoid modifying the original object

        grid_crystal = System_state.grid_crystal
        num_event = System_state.num_event
        self.trans_absorbing_states(idx,System_state,sites_occupied)
        
        self.transition_matrix()
        self.markov_matrix()
        self.absorption_probability_matrix()
        self.calculate_transition_rates_absorbing_states(num_event)
        self.calculate_superbasin_environment(grid_crystal)
        
   
    def trans_absorbing_states(self,start_idx,System_state,sites_occupied):
        
        stack = [start_idx]
        visited = set()

        grid_crystal = System_state.grid_crystal
        self.absorbing_states = []
        self.transient_states_transitions = []
        self.absorbing_states_transitions = []
        aux_transitions = []

        while stack:
            idx = stack.pop()

            if idx not in visited:
                site = grid_crystal[idx]
                is_absorbing = True # Assume that it is an absorbing state
                
                for transition in site.site_events:
                    transition_with_idx = transition + [idx]
                    if transition[3] < self.E_min:

                        is_absorbing = False # If it is a easy way out, it is a transient state (< E_min)
                        if transition_with_idx not in self.transient_states_transitions:
                            self.transient_states_transitions.append(transition_with_idx)
                    else:
                        aux_transitions.append(transition_with_idx)
                    
                if is_absorbing:
                    self.absorbing_states.append(idx)

                else:
                    for transition in site.site_events:
                        # Visit all the transitions from a transient state, even those
                        # with larger Act. Energy than E_min
                        if transition[1] not in visited:
                            stack.append(transition[1])
                
                # Control of the states visited
                visited.add(idx)
                                
            # Virtual migrations to calculate activation energies
            if stack:
                System_state.processes((transition[0], stack[-1], transition[2], idx))
                
    
        System_state.processes((transition[0], start_idx, transition[2], idx)) 

        # Construct the transitions to the absorbing states
        for absorbing_state in self.absorbing_states: 
            i = 0
            while i < len(aux_transitions):
                transition = aux_transitions[i]
                if transition[1] == absorbing_state:
                    self.absorbing_states_transitions.append(transition)
                    aux_transitions.pop(i)  # Remove the processed transition
                else:
                    i += 1
                
                
        self.transient_states = list({transition[-1] for transition in self.transient_states_transitions})

        self.superbasin_idx = self.absorbing_states + self.transient_states 
        
        # Check that grid_crystal is in the original state
        self.verify_grid_crystal(System_state,sites_occupied) 
   
    def transition_matrix(self):
        
        transitions = self.absorbing_states_transitions + self.transient_states_transitions
        
        # Create a dictionary for quick lookup
        transition_dict = {(transition[1], transition[-1]): transition[0] 
                           for transition in transitions}
        # Initialize the transition matrix with zeros
        n = len(self.superbasin_idx)
        A_transitions = np.zeros((n, n))
        
        state_origins = self.superbasin_idx
        state_destinations = self.superbasin_idx
        
        # Fill the diagonal elements
        for i, state_origin in enumerate(state_origins):
            
            if state_origin in self.absorbing_states:
                A_transitions[i, i] = 0
            else:
                tau = sum(transition[0] for transition in transitions if 
                          state_origin == transition[-1])
                A_transitions[i, i] = tau
                
        # Fill the off-diagonal elements
        for i, state_origin in enumerate(state_origins):
            for j, state_destination in enumerate(state_destinations):
                if i != j:
                    rate = -transition_dict.get((state_origin, state_destination), 0)
                    A_transitions[i, j] = max(rate, 0)
                    
        self.A_transitions = A_transitions
        
    def transition_matrix_2(self):
        self.A_transitions_2 = []
        transitions = self.absorbing_states_transitions + self.transient_states_transitions

        # Create a dictionary for quick lookup
        transition_dict = {}
        for transition in transitions:
            key = (transition[1], transition[-1])
            transition_dict[key] = transition[0]

        for i,state_origin in enumerate(self.superbasin_idx):
            row = [] # Each row include the transition from each state_origin
            for j,state_destination in enumerate(self.superbasin_idx):
                
                # Aii = 0 for the absorbing border states
                
                if i == j:
                    if state_origin in self.absorbing_states:
                        row.append(0)
                    else:
                        # For the diagonal in transient states: sum runs over all k superbasin states
                        tau = sum(transition[0] for transition in transitions if
                                  state_origin == transition [-1])
                        row.append(tau)
                        
                else:
                    
                    rate = -transition_dict.get((state_origin, state_destination), 0)
                    rate = rate if rate > 0 else 0
                    row.append(rate)
                    
            self.A_transitions_2.append(row)
        self.A_transitions_2 = np.array(self.A_transitions_2)
    
     
    def markov_matrix(self):
        
        self.M_Markov = []
        transitions = self.absorbing_states_transitions + self.transient_states_transitions

        for i,state_origin in enumerate(self.superbasin_idx):
            row = [0] * len(self.superbasin_idx)  # Initialize row with zeros
            
            if state_origin in self.absorbing_states:
                row[i] = 1 # Absorbing states have a self-transition probability of 1
                
            else:
                for transition in transitions:
                    origin, destination, rate = transition[-1], transition[1], transition[0]
                    if state_origin == origin:
                        j = self.superbasin_idx.index(destination)
                        row[j] = rate / self.A_transitions[i][i]                  
                    
            self.M_Markov.append(row)
        self.M_Markov = np.array(self.M_Markov)
      
         
    def absorption_probability_matrix(self):
        
        self.B_absorption = []
        
        n_transient = len(self.transient_states)
        n_absorbing = len(self.absorbing_states)
        

        T_transient = self.M_Markov[n_absorbing:,n_absorbing:]# Transient matrix
        R_recurrent = self.M_Markov[n_absorbing:,:n_absorbing] # Recurrent matrix
        
        # Calculate the fundamental matrix N
        I = np.eye(n_transient)
        # self.N = np.linalg.inv(I - T_transient)
        self.N = np.linalg.pinv(I - T_transient) # Pseudo-inverse 
        
        # Calculate the absorption probabilities matrix B
        self.B_absorption = np.dot(self.N, R_recurrent) 
        
      
    def calculate_transition_rates_absorbing_states(self,num_event,T=300):
        
        # Mean time to absorbing state j from the transient state i
        # (first passage time - FPT)
        FPT = np.sum(self.N, axis=1)
        # Calculate transition_rates
        transition_rates = self.B_absorption / FPT[:, None]
        
        # We are only interested in the absorbing states
        # Not necessary the probability from each transient state to each absorbing state
        sum_transition_rates = np.sum(transition_rates,axis=0)
        self.transition_rates = np.where(sum_transition_rates < 0, 0, sum_transition_rates)
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E12;  # nu0 (s^-1) bond vibration frequency
        self.EAct = -kb*T*np.log(self.transition_rates/nu0)
        
        self.site_events_absorbing = [
            (transition_r, absorbing_state, num_event - 2, EAct, self.particle_idx)
            for transition_r, absorbing_state, EAct 
            in zip(self.transition_rates, self.absorbing_states, self.EAct)
            ]
  
    def calculate_superbasin_environment(self,grid_crystal):
        
        self.superbasin_environment = set()
        
        for idx in self.superbasin_idx:
            
            self.superbasin_environment = self.superbasin_environment.union(grid_crystal[idx].supp_by)
            
        self.superbasin_environment = set(list(self.superbasin_environment) + self.superbasin_idx)
        
        self.superbasin_environment.discard('Substrate') 

    # Verify that we leave grid_crystal in the original state
    def verify_grid_crystal(self,System_state,sites_occupied):       
        
        for site in sites_occupied:
            # It should be occupied, but it is not
            if System_state.grid_crystal[site].chemical_specie != System_state.chemical_specie:
                # Select deposition event
                event = System_state.grid_crystal[site].site_events[0]
                # Remove the site from sites_occupied
                System_state.sites_occupied.remove(event[1])
                # Introduce the particle
                System_state.processes((event[0], event[1], event[2], event[1])) 

        
      