# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""




import cProfile
import sys
from initialization import initialization,save_variables,search_superbasin
from KMC import KMC
import numpy as np
import time
import platform


save_data = True
lammps_file = True



def main():



    for n_sim in range(0,1):
        
        System_state,rng,paths,Results = initialization(n_sim,save_data,lammps_file)
    
        System_state.add_time()
            
        System_state.plot_crystal(45,45,paths['data'],0)    
        j = 0
        
        snapshoots_steps = int(1e0)
        starting_time = time.time()
    # =============================================================================
    #     Deposition
    # 
    # =============================================================================
        if System_state.experiment == 'deposition':   
    
            nothing_happen = 0
            # list_time_step = []
            list_sites_occu = []
            thickness_limit = 10 # (1 nm)
            System_state.measurements_crystal()
            i = 0
            while System_state.thickness < thickness_limit:
                i+=1
          
                System_state,KMC_time_step, _ = KMC(System_state,rng)
                                
                list_sites_occu.append(len(System_state.sites_occupied))
                
                if np.mean(list_sites_occu[-System_state.n_search_superbasin:]) == len(System_state.sites_occupied):
                # if np.mean(list_time_step[-System_state.n_search_superbasin:]) <= System_state.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if System_state.E_min - System_state.energy_step > 0:
                        System_state.E_min -= System_state.energy_step
                    else:
                        System_state.E_min = 0
                
                if System_state.n_search_superbasin == nothing_happen:
                    search_superbasin(System_state)
                elif nothing_happen> 0 and nothing_happen % System_state.n_search_superbasin == 0:
                    if System_state.E_min_lim_superbasin >= System_state.E_min + System_state.energy_step:
                        System_state.E_min += System_state.energy_step
                    else:
                        System_state.E_min = System_state.E_min_lim_superbasin
                    search_superbasin(System_state)
                    
    
                    
                # print('Superbasin E_min: ',System_state.E_min)
            
                if i%snapshoots_steps== 0:
                    System_state.add_time()
                    
                    j+=1
                    System_state.measurements_crystal()
                    print(str(System_state.thickness/thickness_limit * 100) + ' %','| Thickness: ', System_state.thickness, '| Total time: ',System_state.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                      System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                      System_state.surf_roughness_RMS,end_time-starting_time)
        
                    System_state.plot_crystal(45,45,paths['data'],j)
                    
                    # print('j = ',j)
                    # if j == 5:
                    #     sys.exit()
                        
    
    
    # =============================================================================
    #     Annealing  
    #            
    # =============================================================================
        elif System_state.experiment == 'annealing':
            i = 0
            total_steps = int(5e6)
            nothing_happen = 0
            # total_steps = int(10000)
            System_state.measurements_crystal()
            list_time_step = []
    
            while j*snapshoots_steps < total_steps:
    
                i+=1
                System_state,KMC_time_step, _ = KMC(System_state,rng)
                list_time_step.append(KMC_time_step)
                
    # =============================================================================
    #                 Search of superbasin
    # =============================================================================
                if np.mean(list_time_step[-System_state.n_search_superbasin:]) <= System_state.time_step_limits:
                # if np.mean(list_time_step[-4:]) <= System_state.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if System_state.E_min - System_state.energy_step > 0:
                        System_state.E_min -= System_state.energy_step
                    else:
                        System_state.E_min = 0
                        
                if System_state.n_search_superbasin == nothing_happen:
                    search_superbasin(System_state)
                elif nothing_happen > 0 and nothing_happen % System_state.n_search_superbasin == 0:
                    if System_state.E_min_lim_superbasin >= System_state.E_min + System_state.energy_step:
                        System_state.E_min += System_state.energy_step
                    else:
                        System_state.E_min = System_state.E_min_lim_superbasin
                    search_superbasin(System_state)
                    
    # =============================================================================
    #                     Finish search superbasin
    # =============================================================================
                
                if i%snapshoots_steps== 0:
                    
                    System_state.sites_occupied = list(set(System_state.sites_occupied))
                                        
                    System_state.add_time()
                    j+=1
                    System_state.measurements_crystal()
                    print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                      System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                      System_state.surf_roughness_RMS,end_time-starting_time)
                        
                    System_state.plot_crystal(45,45,paths['data'],j)
                    
    # =============================================================================
    #     Devices: PZT, memristors  
    #            
    # =============================================================================
                    
        elif System_state.experiment == 'ECM memristor':
            
            solve_Poisson = System_state.poissonSolver_parameters['solve_Poisson']
            save_Poisson = System_state.poissonSolver_parameters['save_Poisson']
       
            # Dolfinx only works in Linux
            if solve_Poisson and platform.system() == 'Linux':
                from mpi4py import MPI
                from PoissonSolver import PoissonSolver
        
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                
                mesh_file = System_state.poissonSolver_parameters['mesh_file']
                
                # Initialize Poisson solver on all MPI ranks
                poisson_solver = PoissonSolver(mesh_file,System_state.poissonSolver_parameters, structure=System_state.structure,path_results = paths["results"])
                poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)  # Set appropriate BCs
                
                poisson_solve_frequency = System_state.poissonSolver_parameters['poisson_solve_frequency']  # Solve Poisson every N KMC steps
                
                
            else:
                rank = 0
                comm = None
            
    
            
            i = 0
            #total_steps = int(1e4)
            total_steps = int(1e2)
            # list_sites_occu = []
            
    
            while j*snapshoots_steps < total_steps:
               
                # KMC step runs in serial (only on rank 0)
                if rank == 0:                
                    # Get charge locations and charges from System_state
                    particle_locations, charges = System_state.extract_particles_charges()
                    gen_site_locations = System_state.extract_generation_site_location()
                    
                    if len(particle_locations) > 0 : # In case there is no particles
                      E_field_points = np.concatenate([particle_locations,gen_site_locations],axis = 0)
                    else:
                      E_field_points = gen_site_locations
                else:
                    # Other ranks wait
                    particle_locations = None
                    charges = None
                    gen_site_locations = None
                    E_field_points = None
                
                # Broadcast charge information to all MPI ranks
                if comm is not None:
                    particle_locations = comm.bcast(particle_locations, root=0)
                    charges = comm.bcast(charges, root=0)
                    gen_site_locations = comm.bcast(gen_site_locations, root=0)
                    E_field_points = comm.bcast(E_field_points, root=0)
                  
                comm.Barrier()
                
                if solve_Poisson and platform.system() == 'Linux': 
                  
                  if i%poisson_solve_frequency == 0:
                    
                        run_start_time = MPI.Wtime()
                        uh = poisson_solver.solve(particle_locations,charges)
                        
                        run_time = MPI.Wtime() - run_start_time
                        
                        if rank == 0: print(f'Run time to solve Poisson: {run_time}')
                        E_field = poisson_solver.evaluate_electric_field_at_points(uh,E_field_points)
            
                        if save_Poisson:
                          poisson_solver.save_potential(uh,System_state.time,j+1)
                          
                        if rank == 0: print(f"Poisson solved at step {i}")

                        
                        
                  if rank == 0:
                          
                    """
                    Encapsulate this in a method in crystal_lattice
                    """
                    # Update System_state based on electric field
                    for site, E_site_field in zip(System_state.sites_occupied + System_state.adsorption_sites,E_field):
                      #print(System_state.grid_crystal[site].position,E_site_field)
                      System_state.grid_crystal[site].transition_rates(E_site_field = E_site_field, migration_pathways = System_state.migration_pathways)
                      

                    
                # kMC steps after solving Poisson equation, calculating the electric field and the impact in the transition rates
                if rank == 0:      
                  System_state,KMC_time_step, chosen_event = KMC(System_state,rng)   
                    
                # Synchronize before continuing
                if comm is not None:
                  comm.Barrier()
                     
                i+=1
                if i%snapshoots_steps== 0:
                
                    j+=1
                    # Continue with serial processing on rank 0
                    if rank == 0:
                        System_state.add_time()
    
                        # System_state.measurements_crystal()
                        print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                        end_time = time.time()
                        # if save_data:
                            # Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                            #                               System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                            #                               System_state.surf_roughness_RMS,end_time-starting_time)
                            
                        System_state.plot_crystal(45,45,paths['data'],j)
                        
                        
                              
    
    
        if rank == 0:
          # Variables to save
          variables = {'System_state' : System_state}
          filename = 'variables'
          if save_data: save_variables(paths['program'],variables,filename)
    
        

    
        return System_state

if __name__ == '__main__':
    System_state = main()
# Use cProfile to profile the main function
#     cProfile.run('main()', 'profile_output.prof')    

# import pstats

# # Load and analyze the profiling results
# p = pstats.Stats('profile_output.prof')
# p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions