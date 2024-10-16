# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:35:37 2024

@author: samuel.delgado
"""

from pymatgen.ext.cod import COD
#from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
import matplotlib.pyplot as plt
import numpy as np

def plot_lattice_points(positions_cartesian,azim = 60,elev = 45):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x,y,z = zip(*positions_cartesian)
    
    ax.scatter3D(x, y, z, c='blue', marker='o')
    ax.set_aspect('equal', 'box')
    ax.view_init(azim=azim, elev = elev)

    ax.set_xlabel('x-axis (nm)')
    ax.set_ylabel('y-axis (nm)')
    ax.set_zlabel('z-axis (nm)')
    
    plt.show()
    
    
# Initialize COD with the database URL
cod = COD()


Co_structure = cod.get_structure_by_id(9008492)
# Create the crystal with dimensions approximately 10 x 10 x 2 nm
nx = 1 # Unit cells
ny = 1 # Unit cells
nz = 1 # Unit cells

lattice = Co_structure.lattice
# Convert target dimensions to Angstroms (1 nm = 10 Å)
dimensions = [nx, ny, nz]
scaling_factors = [int(dimensions[i] / lattice.abc[i]) + 1 for i in range(3)]

lattice_model = Co_structure.make_supercell(dimensions)

rotation_matrix = np.array([
    [1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
])

symm_op = SymmOp.from_rotation_and_translation(rotation_matrix, [0, 0, 0])


# # Apply the rotation to the structure
# rotated_structure = Co_structure.copy()
# rotated_structure.apply_operation(symm_op)

# # Apply the CubicSupercellTransformation
# transformation = CubicSupercellTransformation(min_length=20.0)
# cubic_structure = transformation.apply_transformation(rotated_structure)



positions_cartesian = [site.coords for site in Co_structure]

plot_lattice_points(positions_cartesian)