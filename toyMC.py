import time
import pandas as pd
import h5py
import json
import numpy as np
from geometry import *

""" 
TOY MC simulator 
-> generates photosensors in a cylinder surface.
-> generates photons from lists of origins and direction vectors.
-> counts the number of photon interactions per photosensor doing ray tracing.
-> stores the result in h5 format.
"""

stime = time.perf_counter()
json_filename = 'cyl_geom_config.json'

# Generate detector (photsensor placements)
detector = generate_detector(json_filename)
N_photosensors = len(detector.all_points)

# Generate cylinder points
xs, ys, zs = generate_dataset_point_grid(json_filename)

# 0 -> isotropic; 1 -> cherenkov.
SIM_MODE = 0

# iterate over all 3D gird points
Ndatasets = 1 #len(xs) 

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
    h5_evt_hit_IDs_max = np.zeros(maxNhits)#f_outfile.create_dataset("hit_pmt",          shape=(maxNhits,),      dtype=np.int32)
    h5_evt_hit_Qs_max  = np.zeros(maxNhits)#f_outfile.create_dataset("hit_charge",       shape=(maxNhits,),      dtype=np.float32)
    h5_evt_hit_Ts_max  = np.zeros(maxNhits)#f_outfile.create_dataset("hit_time",         shape=(maxNhits,),      dtype=np.float32)

    # counts in every photosensor ID; the array index corresponds to the event and photosensor ID
    # evt_displays     = np.zeros((Nevents,N_photosensors))

    # # define all fields for h5 output file.
    # # actually many of this will not be used (stored) depending on the SIM_MODE
    # # One could pre-define the size of all of this arrays to not use .append... but it seems a sub-leading issue in the computation time.
    # evt_ids          = []  # a unique idenfitifier for the event (actually in the end the datasets have only 1 event)
    # evt_origins      = []  # the origin position for every event.
    # evt_displays     = []  # the event display (counts in every photosensor ID; the array index corresponds to the ID).
    # event_hits_index = []  # the last index in the list of PMT IDs.





    # trk_origins = []  # If we are using tracks, instead of points to simulate the photons, we can store that info here. 
    # trk_dirs    = []  # If we are using tracks, instead of points to simulate the photons, we can store that info here.
    # trk_lengths = []  # If we are using tracks, instead of points to simulate the photons, we can store that info here.
    
    # ray_origins = []  # A list of all the ray origins in one event. (just in case we need it...)
    # ray_dirs    = []  # A list of all the ray dirs in one event.    (just in case we need it...)

    # -- the dataset generation:
    pre_idx = 0
    for i_evt in range(Nevents):
        
        #ID_to_PE = np.zeros(N_photosensors)
        h5_evt_ids[i_evt] = i_evt
        h5_evt_pos[i_evt] = [xs[ds_id], ys[ds_id], zs[ds_id]]

        for i_trk in range(Ntrk):
            Nphot = 5000
            ray_direction = ray_origins = None

            if SIM_MODE == 0:
                ray_vectors = generate_isotropic_random_vectors(Nphot)
                ray_origins = [[xs[ds_id], ys[ds_id], zs[ds_id]] for _ in range(Nphot)]

            elif SIM_MODE == 1:
                ray_vectors = generate_vectors_on_cone_surface(track_direction, np.radians(40), Nphot)
                ray_origins = [[xs[ds_id], ys[ds_id], zs[ds_id]] for _ in range(Nphot)]

            good_indices = check_hits_vectorized_per_track_torch(np.array(ray_origins, dtype=np.float32),\
                                                                 np.array(ray_vectors, dtype=np.float32), \
                                                                 detector.S_radius, \
                                                                 np.array(detector.all_points,dtype=np.float32))

            idx, cts = np.unique(good_indices, return_counts=True)
            Nhits += len(idx)
            h5_evt_hit_idx[i_evt] = Nhits

            print(idx)

            h5_evt_hit_IDs_max[pre_idx:Nhits] = idx
            h5_evt_hit_Qs_max [pre_idx:Nhits] = cts
            h5_evt_hit_Ts_max [pre_idx:Nhits] = np.zeros(len(cts))
            pre_idx = Nhits

    #     #evt_displays.append(ID_to_PE)
        
    # h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt",          shape=(Nhits,),      dtype=np.int32)
    # h5_evt_hit_Qs  = f_outfile.create_dataset("hit_charge",       shape=(Nhits,),      dtype=np.float32)
    # h5_evt_hit_Ts  = f_outfile.create_dataset("hit_time",         shape=(Nhits,),      dtype=np.float32)


    # pre_idx = 0
    # for i in range(Nevents):

    #     non_null_indices = np.where(evt_displays[i_evt][idx] != 0)[0]
    #     non_null_PMT_Q = evt_displays[i][non_null_indices]

    #     print(non_null_indices)

    #     h5_evt_hit_IDs[pre_idx:h5_evt_hit_idx[i]] = non_null_indices
    #     h5_evt_hit_Qs[pre_idx:h5_evt_hit_idx[i]]  = non_null_PMT_Q
    #     h5_evt_hit_Ts[pre_idx:h5_evt_hit_idx[i]]  = np.zeros(len(non_null_PMT_Q))
    #     pre_idx = h5_evt_hit_idx[i]

    h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt",          shape=(Nhits,),      dtype=np.int32)
    h5_evt_hit_Qs  = f_outfile.create_dataset("hit_chargeX",       shape=(Nhits,),      dtype=np.float32)
    h5_evt_hit_Ts  = f_outfile.create_dataset("hit_time",         shape=(Nhits,),      dtype=np.float32)

    #print(h5_evt_hit_IDs_max[0:10])

    h5_evt_hit_IDs[0:Nhits] = h5_evt_hit_IDs_max[0:Nhits]
    h5_evt_hit_Qs[0:Nhits]  = h5_evt_hit_Qs_max [0:Nhits]
    h5_evt_hit_Ts[0:Nhits]  = h5_evt_hit_Ts_max [0:Nhits]

    print(h5_evt_hit_Qs)    

    f_outfile.close()

    #del evts, evts_df, 
    #del evt_ids, trk_origins, trk_dirs, trk_lengths, ray_origins, ray_dirs, evt_displays
#-----------------------------

# store geom information
geom = {
    'positions': list(detector.all_points),
    'case': list(detector.ID_to_case.values()),
}

geom_df = pd.DataFrame(geom)
geom_df.to_hdf('datasets/sim_mode_'+str(SIM_MODE)+'_dataset_geom.h5',key='data', mode='w')
#-----------------------------

print('Torch exec. time: ', f"{time.perf_counter()-stime:.2f} s.")




