# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:13:00 2024

@author: samuel.delgado
"""

import cv2
import os

image_folder = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Tests\Sim_1\Crystal evolution'
video_name = r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Copper deposition\Simulations\Tests\Sim_1\crystal_2.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 24, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()