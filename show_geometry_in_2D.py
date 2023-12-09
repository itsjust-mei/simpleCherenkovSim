import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *
from tools.visualization import *


# Set default values
default_json_filename = 'cyl_geom_config.json'
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')

args = parser.parse_args()

# Access the values
json_filename = args.json_filename

cyl_center, cyl_axis, cyl_radius, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(json_filename)
detector = generate_detector(json_filename)


ID_to_PE = np.zeros(len(detector.all_points))
ID_to_position = {i:x for i,x in enumerate(detector.all_points)}
ID_to_case = detector.ID_to_case
ID_to_PE = {i:x for i,x in enumerate(ID_to_PE)}
# -----------------------------

# do the 2D plot.
show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height)#, file_name='evt_example.pdf')







