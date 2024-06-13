import time
import pandas as pd
import h5py
import json
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *

""" 
TOY MC simulator 
-> generates photosensors in a cylinder surface.
-> generates photons from lists of origins and direction vectors.
-> counts the number of photon interactions per photosensor doing ray tracing.
-> stores the result in h5 format.
"""

stime = time.perf_counter()

# Set default values
default_SIM_MODE = 0
default_json_filename = 'cyl_geom_config.json'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--sim_mode', type=int, default=default_SIM_MODE, help='Sim Mode: 0 -> isotropic; 1 -> cherenkov.')
parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')

args = parser.parse_args()

# Access the values
SIM_MODE = args.sim_mode
json_filename = args.json_filename

# Generate detector (photsensor placements)
detector = generate_detector(json_filename)
N_photosensors = len(detector.all_points)

# Generate cylinder points
xs, ys, zs = generate_dataset_point_grid(json_filename)

# iterate over all 3D gird points



Ndatasets = 10#len(xs) 
for ds_id in range(Ndatasets):
    
    print('Generating dataset: ', ds_id)
    
    Nhits = 0 # this is a counter used to keep track on how many hits we have filled in total for every event.
    # these are just placeholders, in case we want to have more sophisticated datasets in the future.
    Nevents = 1
    Ntrk    = 1

    # create the output h5 file and define some fields we can start filling.
    f_outfile = h5py.File('datasets/sim_mode_'+str(SIM_MODE)+'_dataset_'+str(ds_id)+'_events.h5', 'w')
    h5_evt_ids     = f_outfile.create_dataset("evt_id",           shape=(Nevents,),    dtype=np.int32)
    h5_evt_pos     = f_outfile.create_dataset("positions",        shape=(Nevents,1,3), dtype=np.float32)
    h5_evt_hit_idx = f_outfile.create_dataset("event_hits_index", shape=(Nevents,),    dtype=np.int64)

    # we don't know the number of hits beforehand. We allocate the max number of hits, then at the end we resize.
    maxNhits = Nevents*N_photosensors
    h5_evt_hit_IDs_max = np.zeros(maxNhits)
    h5_evt_hit_Qs_max  = np.zeros(maxNhits)
    h5_evt_hit_Ts_max  = np.zeros(maxNhits)

    # -- the dataset generation:
    pre_idx = 0
    for i_evt in range(Nevents):
        h5_evt_ids[i_evt] = i_evt
        h5_evt_pos[i_evt] = [xs[ds_id], ys[ds_id], zs[ds_id]]

        for i_trk in range(Ntrk):
            Nphot = 10000
            ray_direction = ray_origins = None

            if SIM_MODE == 0:
                ray_vectors = generate_isotropic_random_vectors(Nphot)
                ray_origins = np.ones((Nphot, 3)) * [xs[ds_id], ys[ds_id], zs[ds_id]]

            elif SIM_MODE == 1:
                track_direction = generate_isotropic_random_vectors(1)[0]
                ray_vectors = generate_vectors_on_cone_surface(track_direction, np.radians(40), Nphot)
                ray_origins = np.ones((Nphot, 3)) * [xs[ds_id], ys[ds_id], zs[ds_id]]

            elif SIM_MODE == 2:
                track_direction = generate_isotropic_random_vectors(1)[0]
                ray_vectors = generate_vectors_on_cone_surface(track_direction, np.radians(40), Nphot)
                ray_origins = np.ones((Nphot, 3)) * [xs[ds_id], ys[ds_id], zs[ds_id]] + np.random.uniform(0, 1, (Nphot, 1))*track_direction

            elif SIM_MODE == 3:
                track_direction = np.array([0,1,1])
                ray_vectors = generate_vectors_on_cone_surface(track_direction, np.radians(40), Nphot)
                ray_origins = np.ones((Nphot, 3)) * [0, 0, 0] + 0.2*ds_id*track_direction

            good_indices = check_hits_vectorized_per_track_torch(np.array(ray_origins, dtype=np.float32),\
                                                                 np.array(ray_vectors, dtype=np.float32), \
                                                                 detector.S_radius, \
                                                                 np.array(detector.all_points,dtype=np.float32))

            idx, cts = np.unique(good_indices, return_counts=True)
            Nhits += len(idx)
            h5_evt_hit_idx[i_evt] = Nhits
            h5_evt_hit_IDs_max[pre_idx:Nhits] = idx
            h5_evt_hit_Qs_max [pre_idx:Nhits] = cts
            h5_evt_hit_Ts_max [pre_idx:Nhits] = np.zeros(len(cts))
            pre_idx = Nhits

    h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt",          shape=(Nhits,),      dtype=np.int32)
    h5_evt_hit_Qs  = f_outfile.create_dataset("hit_charge",       shape=(Nhits,),      dtype=np.float32)
    h5_evt_hit_Ts  = f_outfile.create_dataset("hit_time",         shape=(Nhits,),      dtype=np.float32)

    h5_evt_hit_IDs[0:Nhits] = h5_evt_hit_IDs_max[0:Nhits]
    h5_evt_hit_Qs[0:Nhits]  = h5_evt_hit_Qs_max [0:Nhits]
    h5_evt_hit_Ts[0:Nhits]  = h5_evt_hit_Ts_max [0:Nhits]

    f_outfile.close()

print('Total exec. time: ', f"{time.perf_counter()-stime:.2f} s.")




