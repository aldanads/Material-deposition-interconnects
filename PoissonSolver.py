# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:01:44 2025

@author: samuel.delgado
"""

from mpi4py import MPI

import ufl
import numpy as np
import json
from pathlib import Path

from dolfinx import mesh, fem, default_scalar_type, io, geometry
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from dolfinx.io import VTXWriter
import gmsh

from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx import la
import petsc4py.PETSc as PETSc

from scipy.constants import epsilon_0,e

class PoissonSolver():
    
    def __init__(self,mesh_file,poissonSolver_parameters, structure = None, **kwargs):
        
        """
        Initialize the Poisson solver with a mesh file and the structure.
        
        Parameters:
        -----------
         - mesh_file: str
             filename to create the mesh file or to load it
         - structure: Pymatgen structure
             Lattice basis vectors, dimensions of the domain and lattice points with species
         
         **kwargs : dict, optional
           Additional keyword arguments:
           - gmsh_model_rank: int, optional
               the MPI rank that read the .msh file (default: 0)
           - gdim: int, optional 
               geometric dimension of the mesh -> refers to the space in which the mesh is embedded (default: 3)
           - bounding_box_padding: float, optional
               Padding around the lattice points for the bounding box (default: 1.0)
           - mesh_size: float, optional
               Maximum mesh element size (default: 3.0).
        """

        # Set default values for optional parameters
        self.gmsh_model_rank = kwargs.get("gmsh_model_rank", 0)
        self.gdim = kwargs.get("gdim",3)
        self.padding = kwargs.get("bounding_box_padding",5.0)
        self.mesh_size = kwargs.get("mesh_size",0.8) #(Å)
        self.epsilon_gc = kwargs.get("epsilon_gaussian_charge",1.8) #(Å)
        # Set parameters for mesh refinement
        self.active_mesh_refinement = kwargs.get("activate_mesh_refinement",True)
        if self.active_mesh_refinement:
          self.fine_mesh_size = kwargs.get("fine_mesh_size",0.25) #(Å)
          self.refinement_radius = kwargs.get("refinement_radius",1) #(Å)
        
        # Poisson parameters
        self.poissonSolver_parameters = poissonSolver_parameters
        self._calculate_dipole_moment()
        
        mesh_folder = Path("mesh")
        self.mesh_file = mesh_folder / mesh_file
        """
        Generate a mesh if the mesh file does not exist.
        """
        if MPI.COMM_WORLD.rank == 0:
          mesh_folder.mkdir(parents=True, exist_ok=True)  # Create 'mesh' folder if it doesn't exist

          if not self.mesh_file.exists():
            print(f'Rank {MPI.COMM_WORLD.rank}: Starting mesh generation')
            self.generate_mesh(structure)
            print(f'Rank {MPI.COMM_WORLD.rank}: Mesh generation completed')
                      
          else:
            print(f'Rank {MPI.COMM_WORLD.rank}: Using existing mesh file: {self.mesh_file}')
          
        else:
          print(f'Rank {MPI.COMM_WORLD.rank}: Waiting for mesh generation...')
          
        # Barrier: All ranks wait until rank 0 finishes mesh generation
        print(f'Rank {MPI.COMM_WORLD.rank}: Reaching barrier after mesh generation check')
        MPI.COMM_WORLD.Barrier()
        print(f'Rank {MPI.COMM_WORLD.rank}: Passed barrier, proceeding to load mesh')

            
            
        # Load the mesh
        self.domain, self.cell_markers, self.facet_markers = self._load_mesh()
        # Synchronization point for debugging
        MPI.COMM_WORLD.Barrier()
        
        # Example: V ('Lagrange',1) are functions defined in domain, continuous (because Lagrange elements enforce continuity) and degree 1
        self.V = functionspace(self.domain, ('Lagrange',1))
        # Create vector function space for electric field evaluation
        self.V_vec = functionspace(self.domain, ("Lagrange",1, (self.domain.topology.dim,)))
            
        
        # Create facet to cell connectivity required to determine boundary facets 
        self.tdim = self.domain.topology.dim # Get the dimension of the mesh (3D in this case)
        self.fdim = self.tdim - 1 # The dimension of the facets (faces) is one less than the mesh dimension
        self.domain.topology.create_connectivity(self.fdim,self.tdim) # Create connectivity between facets and cells
        self.bcs = []
        
        # Initialize the previous solution for recurrent solution of the Poisson equation
        self.previous_solution = None
        
        # vtx_writer to save the solution
        self.xdmf_writer = None
        self.path_results_folder = kwargs.get("path_results", "")
        self._setup_time_series_output(output_folder="Electric_potential_results")
        
        
        
        
        # ========== Reusable objects ===============
        # Pre-create DGO space for charge density (reused in charge_density())
        # Discontinuous Galerkin (DG) space of order 0 -> no continuity between elements
        # DG(0) useful for defining constant fields, like properties, density
        self.W = fem.functionspace(self.domain, ("DG",0))
        self.rho = fem.Function(self.W,dtype=np.float64)
        
        # Pre-compute cell midpoints (constant for fixed mesh)
        # Get all cell midpoints (for DG0 interpolation)
        #   Local cells (cells owned by this process in parallel computation)
        #   Ghost cells (cells shared between processes in parallel computing)
        num_cells = self.domain.topology.index_map(self.tdim).size_local + self.domain.topology.index_map(self.tdim).num_ghosts
        # Computes the midpoint of each cell in the domain
        #   Since DG(0) functions (like rho) are piecewise constant per element, we typically evaluate them at the midpoint of each element
        self.cell_midpoints = mesh.compute_midpoints(self.domain, self.tdim, np.arange(num_cells, dtype=np.int32))
        self.num_cells = num_cells
        
        # Pre-create trial and test functions
        self.u_trial = ufl.TrialFunction(self.V)
        self.v_test = ufl.TestFunction(self.V)
        
        # Pre-create forms (bilinear form is constant)
        # Pymatgen works in angstrom
        
        # Unit analysis:
        #   grad(u) = V/angstrom
        #   grad(v) = 1/angstrom
        # dx = angstrom^3 
        # a = V * angstrom --> Transform to SI --> Scaling factor 1e-10
        angstrom_to_m = 1e-10
        self.a_form = fem.form(ufl.inner(ufl.grad(self.u_trial), ufl.grad(self.v_test)) * angstrom_to_m * ufl.dx)
        
        # Pre-allocate matrix A (will be reassembled when BCs change)
        # Create empty matrix with correct sparsity pattern
        self.A = fem.petsc.create_matrix(self.a_form)
        
        # Pre-create PETSc vectors
        self.b = fem.petsc.create_vector(fem.form(self.v_test * ufl.dx))
        
        # Track if BCs have changed to trigger matrix reassembly
        self._bcs_changed = True
        
        # Pre-create KSP solver (reuse across solves)
        # Configure KSP solver with initial guess
        self.ksp = PETSc.KSP().create(self.domain.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        self.ksp.getPC().setHYPREType("boomeramg")
        self.ksp.setTolerances(rtol=1e-8,atol=1e-10,max_it=1000)
        self.ksp.setInitialGuessNonzero(True)
        
        # Pre-create solution function
        self.uh = fem.Function(self.V)
        
        # Pre-create E_field function and expression (for evaluate_electric_field_at_points)
        self.E_field = fem.Function(self.V_vec)
        
        # Pre-compute bounding box tree for point evaluation (cached)
        self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        
        
    def _load_mesh(self):
        """
        Load the mesh from the gmsh file.
        """
        
        # All processes load the mesh
        domain, cell_markers, facet_markers = gmshio.read_from_msh(
        str(self.mesh_file), MPI.COMM_WORLD, self.gmsh_model_rank, gdim=self.gdim
        )
        
        return domain, cell_markers, facet_markers
        
    def generate_mesh(self, structure):
          """
          Generate a mesh using GMSH
           - structure: structure variable created by Pymatgen
           - padding: increase margins of the simulation domain (respect to structure)
           - mesh_size: control the coarsen of the mesh
          """
        
          points = np.array([site.coords for site in structure])
          gmsh.initialize()
          gmsh.model.add(self.mesh_file.name)
          
          # Add lattice points to the model using OCC - OpenCASCADE (OCC) geometry module
          point_tags = []
          for point in points:
              tag = gmsh.model.occ.addPoint(point[0], point[1], point[2])
              point_tags.append(tag)
              
          # Create bounding box -> The smallest rectangular (in 2D) or cuboidal (in 3D) volume that fully encloses a given set of objects.
          # Defined by:
          # Minimum coordinates (min_coords) â The smallest (ð¥,ð¦,ð§) values.
          # Maximum coordinates (max_coords) â The largest (ð¥,ð¦,ð§) values.
          min_coords = np.min(points, axis=0) - self.padding
          max_coords = np.max(points, axis=0) + self.padding
          box_tag = gmsh.model.occ.addBox(
              min_coords[0], min_coords[1], min_coords[2], # Start (x, y, z)
              max_coords[0] - min_coords[0], # Width (x-dimension)
              max_coords[1] - min_coords[1], # Height (y-dimension)
              max_coords[2] - min_coords[2] # Depth (z-dimension)
              )
          
          # Synchronizes the OpenCASCADE (OCC) geometry kernel with gmsh model.
          # Necessary after modifying the geometry (adding points, surfaces, volumes) to recognizes the changes
          gmsh.model.occ.synchronize()
          
          # Embed lattice points (point_tags) into the volume (box_tag)
          # gmsh.model.mesh.embed(dim, tags, target_dim, target_tag)
              # dim = 0 â The dimension of the entities being embedded (0 = points).
              # tags = point_tags â A list of point tags (the points to embed).
              # target_dim = 3 â The dimension of the target entity (3 = volume).
              # target_tag = box_tag â The tag of the target volume (where points are embedded).
          gmsh.model.mesh.embed(0,point_tags,self.gdim,box_tag)
          
          # Add physical group for the domain (3D elements)
          domain_tag = gmsh.model.addPhysicalGroup(self.gdim,[box_tag])
          gmsh.model.setPhysicalName(self.gdim,domain_tag,"domain")
          
          # Add physical group for the boundary (2D elements)
          # Extract boundary entities (surfaces) of the 3D box -> boundary of 3D volume box_tag
          boundary_tags = gmsh.model.getBoundary([(self.gdim,box_tag)], oriented = False)
          # Define physical group -> Surface (dim=2) -> Extract the tags of the surfaces
          boundary_tag = gmsh.model.addPhysicalGroup(self.gdim-1, [tag for dim, tag, in boundary_tags])
          gmsh.model.setPhysicalName(self.gdim-1,boundary_tag,"boundary")
          
          
          if self.active_mesh_refinement:
            print('Starting refinement')
            self._add_adaptative_refinement(points)
          else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax",self.mesh_size) # Coarsen globally
            
          # Generate a coarse mesh
          gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Frontal-Delaunay (for efficiency)
          
          gmsh.model.mesh.generate(self.gdim)
          
          # Verify mesh quality
          self._verify_mesh_quality(points)
          # Verify refinement
          #self._verify_refinement_smaller_radii(points)
        
          gmsh.write(str(self.mesh_file))
          gmsh.finalize()
    
    
    def _add_adaptative_refinement(self,site_positions):
      """
      Add distance-based mesh refinement near particles
      Use Ball fields    
      """
      
      try:
        existing_fields = gmsh.model.mesh.field.list()
        for field_id in existing_fields:
          gmsh.model.mesh.field.remove(field_id)
          
      except:
        pass
      
      
      ball_fields = []
      
      total_sites = len(site_positions)
      
      
      for i, pos in enumerate(site_positions):
        # Create distance field to this particle
        ball_field = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(ball_field, 'VIn', self.fine_mesh_size)
        gmsh.model.mesh.field.setNumber(ball_field, 'VOut', self.mesh_size)
        gmsh.model.mesh.field.setNumber(ball_field, 'XCenter', pos[0])
        gmsh.model.mesh.field.setNumber(ball_field, 'YCenter', pos[1])
        gmsh.model.mesh.field.setNumber(ball_field, 'ZCenter', pos[2])
        gmsh.model.mesh.field.setNumber(ball_field, 'Radius', self.refinement_radius)
        

        
        ball_fields.append(ball_field)
        

        # Show the progress
        if total_sites < 20 or i % max(1, total_sites // 10) == 0 or i == total_sites - 1:
          progress = (i + 1) / total_sites * 100
          print(f'Refinement progress: {progress:.1f} % ({i+1}/{total_sites})')
        
      
      if ball_fields:  
        if len(ball_fields) > 1:  
          # Combine all threshold fields using minimum (finest mesh)
          min_field = gmsh.model.mesh.field.add('Min')
          gmsh.model.mesh.field.setNumbers(min_field,'FieldsList', ball_fields)
          gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
          print(f'Combined {len(ball_fields)} ball fields using Min field (ID: {min_field})')

        else:
          gmsh.model.mesh.field.setAsBackgroundMesh(ball_fields[0])
          
          
        
        print(f"Applied {len(ball_fields)} ball refinement fields")
            
      else:
        # Fallback to global mesh size if no refinement fields
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)
        print("No refinement fields applied, using global mesh size")
        
      
    def _verify_mesh_quality(self,site_positions):
      """
      Verify mesh quality and refinement after generation
      """
      print('\n=== Mesh Quality Verification ===')
      
      try:
        #Get mesh statistics
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        
        print(f'Total nodes: {len(nodes[0])}')
        print(f'Element types: {len(elements[0])}')
        
        if len(elements[0]) > 0:
          total_elements = sum(len(elem_tags) for elem_tags in elements[1])
          print(f'Total elements: {total_elements}')
          
          
        # Check mesh size near particles if refinement is active
        #if self.active_mesh_refinement:
        self._check_refinement_near_particles(site_positions)
          
          
      except Exception as e:
        print(f'Mesh verification failed: {str(e)}')
        
      print('=== End Mesh Verification ===\n')
      
    def _check_refinement_near_particles(self,site_positions):
      """
      Check if mesh is actually refined near particle positions
      """
      print('Checking mesh refinement near particles...')
      
      try:
        #Get all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1,3)
        
        coords_min = np.min(node_coords,axis=0)
        coords_max = np.max(node_coords,axis=0)
        domain_size = coords_max - coords_min
        
        # Define margin to avoid boundaries
        margin = domain_size * 0.4
        
        #Filter interior nodes
        interior_mask = (
          (node_coords[:, 0] > coords_min[0] + margin[0]) &
          (node_coords[:, 0] < coords_max[0] - margin[0]) &
          (node_coords[:, 1] > coords_min[1] + margin[1]) &
          (node_coords[:, 1] < coords_max[1] - margin[1]) &
          (node_coords[:, 2] > coords_min[2] + margin[2]) &
          (node_coords[:, 2] < coords_max[2] - margin[2])
        )
        
        interior_nodes = node_coords[interior_mask]
        
        # Check a few particles
        particles_to_check = min(5, len(site_positions))
        
        if self.active_mesh_refinement:
          radius_to_find_nodes = self.refinement_radius
        else:
          radius_to_find_nodes = self.mesh_size  
        
        part_checked = 0 # Particles checked
        i = 0 # idx of particles
        while part_checked < particles_to_check:
        #for i in range(particles_to_check):
          particle_pos = site_positions[i]
          
          
          # Check if particle is in interior region
          is_interior = (
                    (particle_pos[0] > coords_min[0] + margin[0]) and
                    (particle_pos[0] < coords_max[0] - margin[0]) and
                    (particle_pos[1] > coords_min[1] + margin[1]) and
                    (particle_pos[1] < coords_max[1] - margin[1]) and
                    (particle_pos[2] > coords_min[2] + margin[2]) and
                    (particle_pos[2] < coords_max[2] - margin[2])
          )
          
          if not is_interior:
            i+=1
          else:
            i+=1
            part_checked+=1
          
          # Find nodes within refinement radius
          distances = np.linalg.norm(interior_nodes - particle_pos, axis = 1)
          nearby_nodes = interior_nodes[distances <= radius_to_find_nodes]
          
          if len(nearby_nodes) > 0:
            # Estimate local mesh density
            if len(nearby_nodes) > 1:
              #Calculate average distance to nearest neighbors
              from scipy.spatial.distance import pdist
              if len(nearby_nodes) > 1:
                pairwise_distances = pdist(nearby_nodes)
                avg_distance = np.mean(pairwise_distances)
                min_distance = np.min(pairwise_distances[pairwise_distances > 1e-10]) # Avoid zero distances
                
                print(f'Particle {i} at ({particle_pos[0]:.3f}, {particle_pos[1]:.3f}, {particle_pos[2]:.3f}):')
                print(f'  - Nodes within {radius_to_find_nodes:.3f} angstrom: {len(nearby_nodes)}')
                print(f'  - Average node spacing: {avg_distance:.4f} angstrom')
                print(f'  - Minimum node spacing: {min_distance:.4f} angstrom')
                
                # Check if refinement is working
                if self.active_mesh_refinement:
                  if min_distance > self.fine_mesh_size * 2:
                    print(f'  - WARNING: Minimum spacing ({min_distance:.4f}) is larger than expected fine mesh size ({self.fine_mesh_size})')
                    
                  else:
                    print(f'  - Refinement appears to be working correctly')
                    
                else:
                  print(f'  - No refinement applied')
                  
          else:
            print(f'Particle {i}: No nodes found within refinement radius (potential issue)')
            
      except Exception as e:
        print(f'Refinement check failed: {str(e)}')
        
        
    def _verify_refinement_smaller_radii(self, site_positions):
      """
      Verify refinement 
      """
      
      
      # 1. Check what fields exist
      all_fields = gmsh.model.mesh.field.list()
      print(f"Total fields: {len(all_fields)}")
      
      for field_id in all_fields:
        field_type = gmsh.model.mesh.field.getType(field_id)
        print(f'Field {field_id}: {field_type}')
        
        if field_type == "Ball":
          try:
            vin = gmsh.model.mesh.field.getNumber(field_id,'VIn')
            vout = gmsh.model.mesh.field.getNumber(field_id, 'VOut')
            radius = gmsh.model.mesh.field.getNumber(field_id,'Radius')
            print(f'  VIn={vin}, VOut={vout}, Radius={radius}')
          except:
            print(f'  Could not get parameters')
            
      
          
      # 2. Check if background field is set
      # Try to verify Min field is background
      min_fields = [f for f in all_fields if gmsh.model.mesh.field.getType(f) == 'Min']
      if min_fields:
        print(f'Min field exists: {min_fields[0]}')
        #Try to see what it combines
        try:
          field_list = gmsh.model.mesh.field.getNumbers(min_fields[0], 'FieldsList')
          print(f'Combines fields: {field_list}')
        except:
          print('Could not get field list')
      
       
        
      # 3. Check a specific particle location more carefully
      if len(site_positions) > 0:
        test_pos = site_positions[0]
        print(f'\nDetailed check for particle at {test_pos}:')
        
        # Get very close nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1,3)
        
        # Check increasingly smaller radii
        for radius in [self.refinement_radius * 2, self.refinement_radius * 1.5, self.refinement_radius, self.refinement_radius * 0.5]:
          distances = np.linalg.norm(node_coords - test_pos, axis = 1)
          nearby = node_coords[distances <= radius]
          print(f'  Nodes within {radius} angstroms: {len(nearby)}')
          
          if len(nearby) > 1:
            from scipy.spatial.distance import pdist
            valid_dists = pdist(nearby)
            valid_dists = valid_dists [valid_dists > 1e-10]
            if len(valid_dists) > 0:
              min_dist = np.min(valid_dists)
              avg_dist = np.mean(valid_dists)
              print(f'   Min spacing: {min_dist:.4f} angstroms, Avg:{avg_dist:.4f} angstroms')
      
                
    
        
    def update_mesh_parameters(self, padding=None, mesh_size=None):
      """
      Update the padding and mesh size for mesh generation.
        
      Parameters:
      -----------
      padding : float, optional
          New padding value (default: None, keeps current value).
      mesh_size : float, optional
          New mesh size value (default: None, keeps current value).
          
          
      Need to call again: self.generate_mesh(structure)
      The initialization create a mesh. If we change the parameters, we have to generate the mesh again.
      """
      if padding is not None:
        self.padding = padding
      if mesh_size is not None:
        self.mesh_size = mesh_size
        
        
        
    def set_boundary_conditions(self,top_value=0.0, bottom_value=0.0):
        """
        Set Dirichlet boundary conditions on the top and bottom layers.
        """
        
        coords = self.domain.geometry.x # Nx3 array of node coordinates
        # Get min and max in the z-axis (third column)
        min_z = np.min(coords[:, 2])
        max_z = np.max(coords[:, 2])

        def top_boundary(x):
            return np.isclose(x[2],max_z)
        
        def bottom_boundary(x):
            return np.isclose(x[2],min_z)
        
        # Find boundaries in domain where top_boundary returns True
        # - domain: finite element mesh
        # - facet dimension: typically tdim-1
        # - top_boundary: condition for selecting the facets
        # + return boundary_facets_top: NumPy array with indices of facets that satisfy the condition
        boundary_facets_top = mesh.locate_entities_boundary(self.domain,self.fdim,top_boundary)
        boundary_facets_bottom = mesh.locate_entities_boundary(self.domain,self.fdim,bottom_boundary)
        
        # Assign values to u_top and u_bottom at specific points
        u_top = fem.Function(self.V)
        u_top.interpolate(lambda x: np.full_like(x[0],top_value))
        
        u_bottom = fem.Function(self.V)
        u_bottom.interpolate(lambda x: np.full_like(x[0],bottom_value))
        
        # Obtain the degree of freedom (DOFs): the nodes
        boundary_dofs_top = fem.locate_dofs_topological(self.V, self.fdim, boundary_facets_top)
        boundary_dofs_bottom = fem.locate_dofs_topological(self.V, self.fdim, boundary_facets_bottom)
        
        # Apply Dirichlet boundary conditions
        bc_top = fem.dirichletbc(u_top, boundary_dofs_top)
        bc_bottom = fem.dirichletbc(u_bottom, boundary_dofs_bottom)
        
        self.bcs = [bc_top, bc_bottom]
        
        # Mark that BCs have changed
        self._bcs_changed = True
        
        
    def charge_density(self, charge_locations, charges, tolerance = 2):
        """
        Create a DG0 Function representing charge density with Gaussian approximations
        of point charges at specified locations.
        
        Reuses pre-allocated self.rho and self.cell_midpoints
        
        Parameters:
        -----------
        tolerance : float, optional
        Tolerance in % for numerical errors in charge conservation (default: 1)
        """
        
        # Reuse pre-allocated arrays
        cell_values = np.zeros(self.num_cells, dtype = np.float64)
        
        for x0, q in zip(charge_locations, charges):
          # Vectorized Gaussian computation for all cells
          r_sq = np.sum((self.cell_midpoints - x0) ** 2, axis = 1)
          gauss_values = q * np.exp(-r_sq / (2 * self.epsilon_gc ** 2)) / ((2 * np.pi * self.epsilon_gc ** 2) ** (self.tdim / 2))
          
          cell_values += gauss_values
          
        with self.rho.vector.localForm() as local_rho:
          local_rho.setArray(cell_values[:self.W.dofmap.index_map.size_local])
          
        # Sum the local charges across all processes using MPI
        local_charge = fem.assemble_scalar(fem.form(self.rho * ufl.dx))  # Local total charge for this process
    
        # Gather the total charge from all processes
        total_charge = MPI.COMM_WORLD.allreduce(local_charge, op=MPI.SUM)
        
        expected_charge = sum(charges)
        charge_error = 100 * abs((total_charge - expected_charge) / expected_charge)
        
        if charge_error > tolerance:
          if MPI.COMM_WORLD.rank == 0:
            error_msg = (
              f"\nCHARGE CONSERVATION ERROR:\n"
              f"- Total charge: {total_charge:.4e} C\n"
              f"- Expected:     {expected_charge:.4e} C\n"
              f"- Error:        {charge_error:.2f}% of expected charge\n\n"
              f"SOLUTIONS:\n"
              f"1. Increase epsilon (standard deviation) to control how widely the charge is spread with the Gaussian distribution (current: {self.epsilon_gc:.2f}):\n"
              f"   - Larger epsilon: charge spreads out over more cells (reduces singularities but may lose accuracy)\n"
              f"   - Smaller epsilon: charge is more localized (may lead to numerical issues if the mesh is too coarse)\n"
              f"2. Use finer mesh resolution:\n"
              f"   - Smaller cells better resolve point charges (current: {self.mesh_size:.2f})\n"
          )
            #raise ValueError(error_msg)     
            print(error_msg)
            import sys
            sys.stdout.flush()
    
          # Synchronize to ensure rank 0 finishes printing, then abort all processes
          MPI.COMM_WORLD.Barrier()
          MPI.COMM_WORLD.Abort(1)      
            
        #else:
          # Only print if rank = 0
        #  if MPI.COMM_WORLD.rank == 0:
        #    print(f"Total charge validated: {total_charge:.2e} C (Charge error = {charge_error:.2f}% of expected charge)")
          
        return self.rho
        
        
        
    def solve(self, charge_locations, charges, charge_err_tol = 2):
        """
        Solve the Poisson equation with the given charge locations and magnitudes.
        
        Reuses pre-allocated objects instead of creating new ones each call
          - Reuses self.rho (charge density)
          - Reuses self.b (RHS vector)
          - Reassembles self.A only when BCs change (tracked by self._bcs_changed flag)
          - Reuses self.ksp (solver)
          - Reuses self.uh (solution function)
          - Only creates L_form once per solve (not stored as it depends on rho) 
          
        """
        
        # Reassemble matrix only when BCs change
        if self._bcs_changed:
          # Zero out matrix and reassemble with new BCs
          self.A.zeroEntries()
          assemble_matrix(self.A, self.a_form, bcs = self.bcs)
          self.A.assemble()
          self.ksp.setOperators(self.A)
          self._bcs_changed = False # Reset flag
        
        # Create charge density (reuses self.rho internally)
        rho = self.charge_density(charge_locations,charges, charge_err_tol)
        
        
        """
        Unit analysis:
        rho = C/m^3
        epsilon_0 = F/m = C/(m*V)
        dx = m^3
        
        L = V * m
        """
        # Create linear form L (must be recreated as it depends on rho)
        epsilon_r = self.poissonSolver_parameters['epsilon_r']
        L = (rho / (epsilon_0 * epsilon_r)) * self.v_test * ufl.dx
        L_form = fem.form(L)
        
        # Reuse pre-allocated vector self.b --> We zero and reuse
        self.b.zeroEntries()
        assemble_vector(self.b,L_form)
        
        fem.apply_lifting(self.b, [self.a_form], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)
        
        # Reuse solution function self.uh:
        if self.previous_solution is not None:
          self.uh.x.array[:] = self.previous_solution.x.array[:] # Set initial guess
        else:
          self.uh.x.array[:] = 0.0 # Zero initial guest on first solve
        
        # Solve with initial guess
        self.ksp.solve(self.b, self.uh.x.petsc_vec)
        
        #Store solution for next iteration
        if self.previous_solution is None:
          self.previous_solution = fem.Function(self.V)
          
        self.previous_solution.x.array[:] = self.uh.x.array[:]
        
        return self.uh
        

    def evaluate_electric_field_at_points(self,uh,points):
      """
      Evaluate electric field E = -?V at specific points
      
      Parameters:
      -----------
      uh : dolfinx.fem.Function
          Solution of Poisson equation (electric potential)
      points: array-like, shape (n_points, 3)
          Points where to evaluate electric field
          
      Returns:
      --------
      E_values : np.array, shape (n_points, 3)
          Electric field values at given points
      """
      
      # Compute gradient expression (E = -?V)
      E_expr =  -ufl.grad(uh)
      
      # Create expression for evaluation at points
      expr = fem.Expression(E_expr, self.V_vec.element.interpolation_points()) 
      self.E_field.interpolate(expr)

      
      # Convert points to numpy array with correct dtype
      points_array = np.asarray(points,dtype=np.float64)
      if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
        
      # Find cells whose bounding-box collide with the points
      cell_candidates = geometry.compute_collisions_points(self.bb_tree,points_array)
      # Choose one of the cells that contains the point
      colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_array)
      
      # Initialize result array for all points
      E_values_global = np.zeros((len(points_array), 3), dtype=np.float64)
      
      # Collect points and corresponding cells
      local_cells = []
      points_on_proc = []
      local_idx = []
      
      for i, point in enumerate(points_array):
        
        if len(colliding_cells.links(i)) > 0:
          points_on_proc.append(point)
          local_cells.append(colliding_cells.links(i)[0])
          local_idx.append(i)
        
      if len(points_on_proc) > 0:
        points_on_proc = np.array(points_on_proc, dtype = np.float64)
        # Evaluate expression at all points at once
        E_local = self.E_field.eval(points_on_proc, local_cells) # Units: V/Å
        
        
        for j, global_idx in enumerate(local_idx):
          if len(local_idx) == 1:
            E_values_global[global_idx] = E_local
          else:
            E_values_global[global_idx] = E_local[j]
          
      # Communicate results across all MPI processes
      # Each point is evaluated by exactly one process, so we sum the results
      E_values_global = MPI.COMM_WORLD.allreduce(E_values_global, op=MPI.SUM)
             
      return E_values_global * 1e10 * self.bond_polarization_factor # Units: V/m
      
    def test_point_charge_analytical(self,charge_location,charges):
      """
      Test electric field against analytical point charge solution
      """
      epsilon_r = self.poissonSolver_parameters['epsilon_r']
      
      actual_center = self.find_actual_charge_center(charge_location)
            
      uh = self.solve(charge_location,charges)
      
      # Test points at varios distances
      
      test_distances = np.linspace(-0.5,0.5,11)
      
      for r in test_distances:
      
        test_point = np.array([[charge_location[0][0] + r, charge_location[0][1], charge_location[0][2]]])
        E_computed = self.evaluate_electric_field_at_points(uh,test_point)
        E_magnitude = np.linalg.norm(E_computed[0])
        
        # Analytical solution
        E_analytical = self.bond_polarization_factor * charges[0] * (r * 1e-10) / (4 * np.pi * epsilon_0 * epsilon_r * ((r * 1e-10)**2 + (self.epsilon_gc * 1e-10)**2)**1.5)
        
        proportion_E_computed_analytical = abs(E_magnitude / E_analytical)
        
        if MPI.COMM_WORLD.rank == 0:
          print(f"Distance {r} angstrom: Computed={E_magnitude:.2e}, Analytical={E_analytical:.2e}, Proportion E (computed/analytical)={proportion_E_computed_analytical:.2e}")
          print(f"Computed(x,y,z)=({E_computed[0][0]:.2e},{E_computed[0][1]:.2e},{E_computed[0][2]:.2e})")

        
        
      return uh
      
    
    def test_uniform_field(self):
      """
      Test with linear potential (should give uniform field)
      Set up linear boundary conditions that should create uniform field
      For example: V(x) = -E0 * x should give E = E0 in x-direction
      """
      pass
      
      
      
    def evaluate_function_at_points(self,function,points):
      """
      Evaluate a dolfinx function at specific points
      """
      points_array = np.asarray(points,dtype=np.float64)
      
      # Find cells containing points
      bb_tree = geometry.bb_tree(self.domain,self.domain.topology.dim)
      cell_candidates = geometry.compute_collisions_points(bb_tree,points_array)
      colliding_cells = geometry.compute_colliding_cells(self.domain,cell_candidates,points_array)
      
      # Prepare cells array
      cells = []
      for i in range(len(points_array)):
        if len(colliding_cells.links(i)) > 0:  
          cell_id = colliding_cells.links(i)[0]
          cells.append(cell_id)
        else:
          cells.append(0)
          
      # Evaluate function
      values = function.eval(points_array,cells)
      
      return values
      
    
    def find_actual_charge_center(self, requested_center):
      """
      Find where the charge density is actually maximized on the mesh
      """    
      
      # Create a test charge at the requested position
      test_rho = self.charge_density([requested_center],[1.0])
      
      # Evaluate charge density at all midpoints
      rho_values = self.evaluate_function_at_points(test_rho,self.cell_midpoints)
      
      # Find the maximum
      max_idx = np.argmax(rho_values)
      actual_center = self.cell_midpoints[max_idx]
      
      if MPI.COMM_WORLD.rank == 0:
        print(f"Requested center: {requested_center}")
        print(f"Actual center on mesh: {actual_center}")
        print(f"Offset: {actual_center - requested_center}")
        
      return actual_center
      
             

      
    def _calculate_dipole_moment(self):
    
      """
      Padovani, A., Larcher, L., Pirrotta, O., Vandelli, L., & Bersuker, G. (2015). 
      Microscopic modeling of HfO x RRAM operations: From forming to switching. 
      IEEE Transactions on electron devices, 62(6), 1998-2006.
    
      McPherson, J. W., & Mogul, H. C. (1998). Underlying physics of the thermochemical 
      E model in describing low-field time-dependent dielectric       
      breakdown in SiO 2 thin films. Journal of Applied Physics, 84(3), 1513-1523.
      
      
      
      ************* Formula for dielectric moment: *******************
      McPherson, J., J. Y. Kim, A. Shanware, and H. Mogul. "Thermochemical description 
      of dielectric breakdown in high dielectric constant              
      materials." Applied Physics Letters 82, no. 13 (2003): 2121-2123.
      """
      # Dipole moment: Units (eÅ); 1D = Debye	 0.2081943 e·Å (large dipole moment is around 11D)
      # Dipole moment: Units (enm)
      L = {'Tetrahedron': 1/3, 'Octahedron': 1, 'Trigonal': np.sqrt(2/3), 'Cube': np.sqrt(1/3), 'Disheptahedral': np.sqrt(2/3), 'Cuboctahedral': np.sqrt(1/3)}
      
      metal_valence = self.poissonSolver_parameters['metal_valence'] # Metal valence
      d_metal_O = self.poissonSolver_parameters['d_metal_O'] #Units: Å
      chem_env_symmetry = self.poissonSolver_parameters['chem_env_symmetry'] # Symmetry in the local environment (molecule)
      active_dipoles = self.poissonSolver_parameters['active_dipoles']
      epsilon_r = self.poissonSolver_parameters['epsilon_r']
      
      dipole_moment = active_dipoles * (metal_valence / 2) * d_metal_O * L[chem_env_symmetry]
      
      self.bond_polarization_factor = ((2+epsilon_r) / 3) * dipole_moment
      
      
    def _setup_time_series_output(self,output_folder="Electric_potential_results"):
      """Call this at the beginning"""
      results_folder = self.path_results_folder / output_folder
      results_folder.mkdir(exist_ok=True, parents=True)
      
      self.filename = results_folder / "E_potential"
      
      self.timestep_info = []
        
        
    def save_potential(self, uh, time_value, time_step, save_CSV = False):
      
      # Name for ParaView
      uh.name = "ElectricPotential"
      
      filename = f"{self.filename}_{time_step:04d}.vtu" 
      
      with io.VTKFile(self.domain.comm,filename,"w") as vtk:
        vtk.write_function(uh)
        
        
      
      if save_CSV == True:
        # Save mesh coordinates and function values to CSV
        mesh_coordinates = self.domain.geometry.x
        function_values = uh.x.array
        data =  np.column_stack((mesh_coordinates, function_values))
        np.savetxt(self.path_results_folder / "fundamentals.csv", data, delimiter=",", header="x,y,z,value", comments="")
        
        
        
