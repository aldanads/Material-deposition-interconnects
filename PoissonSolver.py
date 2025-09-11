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
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io import VTXWriter
import gmsh

from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx import la
import petsc4py.PETSc as PETSc

from scipy.constants import epsilon_0

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
        self.mesh_size = kwargs.get("mesh_size",2.0)
        self.epsilon_gc = kwargs.get("epsilon_gaussian_charge",2)
        
        # Poisson parameters
        self.poissonSolver_parameters = poissonSolver_parameters
        self._calculate_dipole_moment()
        
        """
        Generate a mesh if the mesh file does not exist.
        """
        mesh_folder = Path("mesh")
        mesh_folder.mkdir(parents=True, exist_ok=True)  # Create 'mesh' folder if it doesn't exist
        self.mesh_file = mesh_folder / mesh_file

        if not self.mesh_file.exists():
            self.generate_mesh(structure)
            
            
        # Load the mesh
        self.domain, self.cell_markers, self.facet_markers = self._load_mesh()

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
        
        
        
        
        
    def _load_mesh(self):
        """
        Load the mesh from the gmsh file.
        """
        
        # All processes load the mesh
        domain, cell_markers, facet_markers = gmshio.read_from_msh(
        str(self.mesh_file), MPI.COMM_WORLD, self.gmsh_model_rank, gdim=self.gdim
        )
        
        # Synchronization point for debugging
        MPI.COMM_WORLD.Barrier()
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
        # Minimum coordinates (min_coords) → The smallest (𝑥,𝑦,𝑧) values.
        # Maximum coordinates (max_coords) → The largest (𝑥,𝑦,𝑧) values.
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
            # dim = 0 → The dimension of the entities being embedded (0 = points).
            # tags = point_tags → A list of point tags (the points to embed).
            # target_dim = 3 → The dimension of the target entity (3 = volume).
            # target_tag = box_tag → The tag of the target volume (where points are embedded).
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
        
        # Generate a coarse mesh
        gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Frontal-Delaunay (for efficiency)
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",self.mesh_size) # Coarsen globally
        
        gmsh.model.mesh.generate(self.gdim)
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
        
        
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
        
        
    def charge_density(self, charge_locations, charges, tolerance = 1):
        """
        Create a DG0 Function representing charge density with Gaussian approximations
        of point charges at specified locations.
        
        Parameters:
        -----------
        tolerance : float, optional
        Tolerance in % for numerical errors in charge conservation (default: 1)
        """
        # Discontinuous Galerkin (DG) space of order 0 -> no continuity between elements
        # DG(0) useful for defining constant fields, like properties, density
        W = fem.functionspace(self.domain,("DG",0)) 
        rho = fem.Function(W, dtype=np.float64) # storing floating-point values -> field variable defined over the entire mesh
        
        # Get all cell midpoints (for DG0 interpolation)
        #   Local cells (cells owned by this process in parallel computation)
        #   Ghost cells (cells shared between processes in parallel computing)
        num_cells = self.domain.topology.index_map(self.tdim).size_local + self.domain.topology.index_map(self.tdim).num_ghosts
        # Computes the midpoint of each cell in the domain
        #   Since DG(0) functions (like rho) are piecewise constant per element, we typically evaluate them at the midpoint of each element
        midpoints = mesh.compute_midpoints(self.domain, self.tdim, np.arange(num_cells, dtype=np.int32))
        
        cell_values = np.zeros(num_cells, dtype = np.float64)
        
        
        for x0, q in zip(charge_locations, charges):
          # Vectorized Gaussian computation for all cells
          r_sq = np.sum((midpoints - x0) ** 2, axis = 1)
          gauss_values = q * np.exp(-r_sq / (2 * self.epsilon_gc ** 2)) / ((2 * np.pi * self.epsilon_gc ** 2) ** (self.tdim / 2))
          cell_values += gauss_values
          
        with rho.vector.localForm() as local_rho:
          local_rho.setArray(cell_values[:W.dofmap.index_map.size_local])
          
        # Sum the local charges across all processes using MPI
        local_charge = fem.assemble_scalar(fem.form(rho * ufl.dx))  # Local total charge for this process
    
        # Gather the total charge from all processes
        total_charge = MPI.COMM_WORLD.allreduce(local_charge, op=MPI.SUM)
        #total_charge = fem.assemble_scalar(fem.form(rho * ufl.dx))
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
          
        return rho
        
        
        
    def solve(self, charge_locations, charges, charge_err_tol = 2):
        """
        Solve the Poisson equation with the given charge locations and magnitudes.
        """
        
        # Create charge density
        rho = self.charge_density(charge_locations,charges, charge_err_tol)
        
        # Define variational problem
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        """
        Pymatgen works in angstrom
        
        Unit analysis:
        grad(u) = V/angstrom
        grad(v) = 1/angstrom
        dx = angstrom^3
        
        a = V * angstrom --> Transform to SI --> Scaling factor 1e-10
        """
        angstrom_to_m = 1e-10 # Scale factor to convert to SI
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * angstrom_to_m * ufl.dx
        
        """
        Unit analysis:
        rho = C/m^3
        epsilon_0 = F/m = C/(m*V)
        dx = m^3
        
        L = V * m
        """
        epsilon_r = self.poissonSolver_parameters['epsilon_r']
        L = (rho / (epsilon_0 * epsilon_r)) * v * ufl.dx
        
        # Assemble matrix and vector
        A = assemble_matrix(fem.form(a), bcs=self.bcs)
        A.assemble()
        b = assemble_vector(fem.form(L))
        fem.apply_lifting(b, [fem.form(a)], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.bcs)
        
        # Initialize solution function
        uh = fem.Function(self.V)
        if self.previous_solution is not None:
          uh.x.array[:] = self.previous_solution.x.array[:] # Set initial guess

        # Configure KSP solver with initial guess
        ksp = PETSc.KSP().create(self.domain.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.getPC().setHYPREType("boomeramg")
        ksp.setTolerances(rtol=1e-8,atol=1e-10,max_it=1000)
        ksp.setInitialGuessNonzero(True) # Enable initial guess
        
        
        # Solve with initial guess
        ksp.solve(b, uh.x.petsc_vec)
        
        #Store solution for next iteration
        if self.previous_solution is None:
          self.previous_solution = fem.Function(self.V)
          
        self.previous_solution.x.array[:] = uh.x.array[:]
        
        return uh
        

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
      
      
      if not hasattr(self,'_E_field_cache') or self._last_uh_id != id(uh):
        # For electric field, we need to compute the gradient
        # Create a vector function to store the gradient
        self._E_field_cache = fem.Function(self.V_vec)
        # Create expression for evaluation at points
        expr = fem.Expression(E_expr, self.V_vec.element.interpolation_points()) 
        self._E_field_cache.interpolate(expr)
        self._last_uh_id = id(uh)
        
      E_field = self._E_field_cache
      
      # Convert points to numpy array with correct dtype
      points_array = np.asarray(points,dtype=np.float64)
      if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
        
      if not hasattr(self, '_bb_tree_cache'):
        # Find the correct cell for each point
        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        
      # Find cells whose bounding-box collide with the points
      cell_candidates = geometry.compute_collisions_points(bb_tree,points_array)
      # Choose one of the cells that contains the point
      colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_array)
      
      # Initialize result array for all points
      num_points = len(points_array)
      E_values_global = np.zeros((num_points, 3), dtype=np.float64)
      
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
        E_local = E_field.eval(points_on_proc, local_cells) # Units: V/�
        
        for j, global_idx in enumerate(local_idx):
          E_values_global[global_idx] = E_local[j]
          
      # Communicate results across all MPI processes
      # Each point is evaluated by exactly one process, so we sum the results
      E_values_global = MPI.COMM_WORLD.allreduce(E_values_global, op=MPI.SUM)
             
      return E_values_global * 1e10 * self.bond_polarization_factor # Units: V/m
      

      
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
      # Dipole moment: Units (e�); 1D = Debye	� 0.2081943 e�� (large dipole moment is around 11D)
      # Dipole moment: Units (enm)
      L = {'Tetrahedron': 1/3, 'Octahedron': 1, 'Trigonal': np.sqrt(2/3), 'Cube': np.sqrt(1/3), 'Disheptahedral': np.sqrt(2/3), 'Cuboctahedral': np.sqrt(1/3)}
      
      metal_valence = self.poissonSolver_parameters['metal_valence'] # Metal valence
      d_metal_O = self.poissonSolver_parameters['d_metal_O'] #Units: �
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
        np.savetxt(results_folder / "fundamentals.csv", data, delimiter=",", header="x,y,z,value", comments="")
        
        
        
