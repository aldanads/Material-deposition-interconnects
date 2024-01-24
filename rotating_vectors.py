# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:31:37 2024

@author: samuel.delgado
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lattpy as lp

def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Finds the angle between two vectors."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

def rotate_vector(vector, axis, theta):
    """
    Rotates a 3D vector around a specified axis.

    Parameters:
    - vector: The 3D vector to rotate.
    - axis: The axis of rotation ('x', 'y', or 'z').
    - theta: The rotation angle in radians.

    Returns:
    The rotated vector.
    """
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")
    
    return np.dot(R, vector)

def plot_basis_vectors(basis_vectors,v):
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    start = [0,0,0]
    
    ax.quiver(start[0],start[1],start[2],basis_vectors[0][0],basis_vectors[0][1],basis_vectors[0][2])
    ax.quiver(start[0],start[1],start[2],basis_vectors[1][0],basis_vectors[1][1],basis_vectors[1][2])
    ax.quiver(start[0],start[1],start[2],basis_vectors[2][0],basis_vectors[2][1],basis_vectors[2][2])
    
    ax.set_xlim([-0.5, 0.5]) 
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_aspect('equal', 'box')
            
    fig = plt.figure(dpi=300)
    ax2 = fig.add_subplot(111, projection='3d')
    
    ax2.quiver(start[0],start[1],start[2],v[0][0],v[0][1],v[0][2])
    ax2.quiver(start[0],start[1],start[2],v[1][0],v[1][1],v[1][2])
    ax2.quiver(start[0],start[1],start[2],v[2][0],v[2][1],v[2][2])
    
    ax2.set_xlim([-0.5, 0.5]) 
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_zlim([-0.5, 0.5])
    
def plot_lattice_points(positions_2):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = [vector[0] for vector in positions_2]
    y = [vector[1] for vector in positions_2]
    z = [vector[2] for vector in positions_2]
    
    ax.scatter3D(x, y, z, c='blue', marker='o',s = 0.1)
    ax.set_aspect('equal', 'box')
    ax.view_init(azim=45, elev=10)
    
    ax.set_xlabel('x-axis (nm)')
    ax.set_ylabel('y-axis (nm)')
    ax.set_zlabel('z-axis (nm)')
    
    plt.show()
    
def plot_slice(nearest_neighbors_2):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Select atoms in plane
    pos = [vector for vector in nearest_neighbors_2 if abs(vector[2]) < 1e-15]
    # pos = [vector for vector in nearest_neighbors_2 if vector[2] < -0.1]
    # pos = [vector for vector in nearest_neighbors_2 if vector[2] > 0.1]
    
    x = [vector[0] for vector in pos]
    y = [vector[1] for vector in pos]
    
    ax.scatter(x,y)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x-axis (nm)')
    ax.set_ylabel('y-axis (nm)')
    
    plt.show()


# Default resolution for figures
plt.rcParams["figure.dpi"] = 300 # Default value of dpi = 300

a = 0.358 # (nm)
b = 0.358 # (nm)
c = 0.358 # (nm)
lattice_constants = (a,b,c)
crystal_size = (1, 1,1)
latt = lp.Lattice.fcc(lattice_constants[0])
latt.add_atom()
latt.add_connections(1)
latt.analyze()
latt.build((crystal_size[0], crystal_size[1],crystal_size[2]))

basis_vectors = latt.vectors
#basis_vectors = [[1,0,0],[0,1,0],[0,0,1]]

# =============================================================================
# fcc - direction [111]
# https://www.chegg.com/homework-help/questions-and-answers/example-fcc-general-111-110-combination-preferred-slip-system-ca-1011-101-fcco-111-110-20-q27488166
# =============================================================================

# Using matrix rotation over z axis, so x and y components
# are the same for u and v 
theta = np.pi/4
u = [list(rotate_vector(vector,'z',theta)) for vector in basis_vectors]

# Using matrix rotation over x axis, we stablish that z components should 
# be equal in the three vectors
theta2 = np.arctan(1/u[0][0])
theta2 = np.arctan(u[2][2]/(u[0][1] - u[2][1]))
v = [list(rotate_vector(vector,'x',theta2)) for vector in u]


plot_basis_vectors(basis_vectors,v)

positions = latt.positions
positions_2 = [list(rotate_vector(vector,'z',theta)) for vector in positions]
positions_2 = [list(rotate_vector(vector,'x',theta2)) for vector in positions_2]

#plot_lattice_points(positions_2)

nearest_neighbors = latt.get_neighbor_positions()
nearest_neighbors_2 = [list(rotate_vector(vector,'z',theta)) for vector in nearest_neighbors]
nearest_neighbors_2 = [list(rotate_vector(vector,'x',theta2)) for vector in nearest_neighbors_2]
#print(nearest_neighbors_2)

plot_slice(nearest_neighbors_2)