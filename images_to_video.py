# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:27:22 2024

@author: samuel.delgado
"""

import imageio.v2 as imageio
import os

# Path to the directory containing your PNG images
image_dir = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Tests\Sim_5 - Not finished\Crystal evolution'

# Output file path for the GIF
output_gif_path = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Tests\Sim_5 - Not finishedt\crystal.gif'

# List all PNG images in the directory
image_files = [file for file in os.listdir(image_dir) if file.endswith('.png')]

# Sort files numerically
image_files = sorted(image_files, key=lambda x: int(x.split('_')[0]))

# Create a list to store images
images = []

i = 0
# Read and append each image to the list
for image_file in image_files:
    i +=1
    print(str(i)+'/'+str(len(image_files)))
    image_path = os.path.join(image_dir, image_file)
    image = imageio.imread(image_path)
    images.append(image)

# Create the GIF
imageio.mimsave(output_gif_path, images, duration=0.2)  # Adjust the duration as needed
