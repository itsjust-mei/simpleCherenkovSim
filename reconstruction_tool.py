import numpy as np
import plotly.graph_objs as go
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.optimize import curve_fit

import seaborn as sns
from tools.geometry import *


geo_file = np.load('/work/kmtsui/wcte/wcsim_e-_Beam_260MeV_20cm_0000.geo.npz', allow_pickle=True)
data_file = np.load('/work/kmtsui/wcte/npz/wcsim_mu-_Beam_900MeV_20cm_0000.npz', allow_pickle=True)

array_names = geo_file.files
arrays = []
for name in array_names:
    arrays.append(geo_file[name])
array_names
tube_no, position, orientation = arrays 

position_data = data_file['position'][0]
track_direction = data_file['direction'][0]

#Digi data
hit_time = data_file['digi_hit_time'] # this gets the pmt digi hit time
digi_hit_pmt = data_file['digi_hit_pmt'] # this gets the pmt digi hit pmt
hit_charge = data_file['digi_hit_charge'] # this gets the pmt digi hit charge

# True data
true_hit_start_time = data_file['true_hit_start_time']
true_hit_start_pos_0 = data_file['true_hit_start_pos'][0]
true_hit_start_pos = data_file['true_hit_start_pos']
true_hit_start_dir = data_file['true_hit_start_dir']
true_hit_time = data_file['true_hit_time']
true_hit_dir = data_file['true_hit_dir']
true_hit_pos = data_file['true_hit_pos']
true_hit_pmt = data_file['true_hit_pmt']

true_hit_parent=data_file['true_hit_parent']
track_id=data_file['track_id']
track_start_position_0 = data_file['track_start_position'][0]
vg = 2.20027795333758801e8*100/1.e9; #rough speed of light in water in cm/ns

all_hit_times = np.concatenate(hit_time)
all_digi_hit=np.concatenate(digi_hit_pmt)
all_charges = np.concatenate(hit_charge)

#Get the particle id 
def particle_idx(data_file):
    track_id=data_file['track_id']
    indices_2d = np.where(track_id[0]== 1)
    print("Indices in 1D array with elements equal to 1:", indices_2d[0][0])
    return indices_2d[0][0]

#Get indices of 
def indices_parent(particle_idx):
    return np.where(true_hit_parent[0]== 1)

def tube_hit_counts(digi_hit_pmt):
    tube_hit_count = {}
    for event_hits in digi_hit_pmt:
        for tube in event_hits:
            if tube in tube_hit_count:
                tube_hit_count[tube] += 1
            else:
                tube_hit_count[tube] = 1

    # Create an array to store the count of hits for each tube
    tube_hit_counts = np.zeros(len(tube_no))
    for tube in tube_no:
        if tube-1 in tube_hit_count:
            tube_hit_counts[tube-1] = tube_hit_count[tube-1]

    return tube_hit_counts


def tube_hit_charges(digi_hit_pmt, hit_charge):
    # Count the charge for each tube
    tube_hit_charge = {}
    for j in range(len(digi_hit_pmt)):
        event_hits = digi_hit_pmt[j]
        for i in range(len(event_hits)):
            tube = event_hits[i]
            charge = hit_charge[j][i]
            if tube in tube_hit_charge:
                    tube_hit_charge[tube] += charge
            else:
                    tube_hit_charge[tube] = charge

    # Create an array to store the count of hits for each tube
    tube_hit_charges = np.zeros(len(tube_no))
    for tube in tube_no:
        if tube-1 in tube_hit_charge:
            tube_hit_charges[tube-1] = tube_hit_charge[tube-1]
    return tube_hit_charges

def find_origin(points , track_direction,  ray_origin) :
    """ For a collection of photons calculate the list of ID of the PMTs that get hit."""
    device = torch.device("cpu")

    # Convert NumPy arrays to PyTorch tensors and move to "mps" device
    ray_origin_torch = torch.tensor(ray_origin, dtype=torch.float32, device=device).unsqueeze(0) 
    track_direction_torch = torch.tensor(track_direction, dtype=torch.float32, device=device).unsqueeze(0) 
    points_torch = torch.tensor(points, dtype=torch.float32, device=device)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_torch - ray_origin_torch[:, None, :]

    # Project all vectors onto the ray direction using einsum
    dot_products_numerator = torch.einsum('ijk,ik->ij', vectors_to_points, track_direction_torch)
    dot_products_denominator = torch.sum(track_direction_torch * track_direction_torch, dim=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, None]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_torch[:, None, :] + t_values[:, :, None] * track_direction_torch[:, None, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances_proj = torch.norm(points_torch - closest_points_on_ray, dim=2)

    # find origin point
    distance = distances_proj/np.tan(np.radians(40))
    origin_ray = closest_points_on_ray - distance[:, :, None]*track_direction_torch[:, None, :]/np.sqrt(dot_products_denominator[:, None])

    
    return origin_ray.cpu().numpy()
