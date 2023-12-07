import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import pandas as pd

from objects.geometry import *
from objects.ray import *
from objects.segment import *
from objects.scene import *
from tools import *

# -- example definitions (evenutally move to parameter file)
cyl_center        = np.array([0, 0, 0])
cyl_axis          = np.array([0, 0, 1]) # warning: visualization only works for [0,0,1]!
cyl_radius        = 6
cyl_height        = 10
cyl_barrel_grid   = [14,48]  # 14 cols and 48 rows
cyl_cap_rings     = [40,34,30,24,20,16,10,5,1] # 9 concentric rings with a number of sensors specified by the array
cyl_sensor_radius = 0.25

# ---------------------
detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)
stime = time.time()
Nevents  = 100
# one event is one track!! (logic is set up accordingly)


# define all fields for h5 output file.
evt_ids     = []  # a unique idenfitifier for the event
trk_origins = []  # the track origin.    This is a list of lists for every event.
trk_dirs    = []  # the track direction. This is a list of lists for every event.
trk_lengths = []  # the track length.    This is a list of lists for every event.
                  # at some point the length will be replaced by something proper (e.g. E, and then we store also PDG)
                  # ... (but lets use this length for the moment).

ray_origins = []
ray_dirs    = []
displays    = []

for nevt in range(Nevents):
    ID_to_PE = {i:0 for i in range(len(detector.all_points))}
    evt_ids.append(nevt)

    # evt_trk_dir    = []
    # evt_origins = []
    # evt_lengths = []

    if nevt % 10 == 0:
        print(nevt)
    for i in range(1):

        # let's chose the vertex uniformely in a 2x2x2 box around the [0,0,0]
        track_origin        = np.array([-1+2*np.random.uniform() for _ in range(3)])

        # let's chose the track direction randomly
        track_direction     = generate_isotropic_random_vector()
        track_direction     = track_direction/np.linalg.norm(track_direction)

        L = 1+1.5*np.random.uniform()

        trk_dirs.append(track_direction)
        trk_origins.append(track_origin)
        trk_lengths.append(L)

        # assume photons proportional to length
        Nphot = int(1000*L)

        # in this example we have only one segment for all the track
        track = Segment(track_origin, track_direction, length=cyl_radius)
        track.gen_ch_rays(np.radians(40), num_samples=Nphot)

        evt_ray_origins = []
        evt_ray_dirs    = []
        for n in range(Nphot):
            vect = [track.ray_vectors[:,0][n],track.ray_vectors[:,1][n],track.ray_vectors[:,2][n]]
            vect = vect/np.linalg.norm(vect)

            origin = (L*np.random.uniform())*track_direction*(n/Nphot) # we assume dNphot/ds is constant...
            sensor_coordinates, dist, sensor_idx = detector.search_nearest_sensor_for_ray(origin,vect)

            if dist < cyl_sensor_radius:
                ID_to_PE[sensor_idx] += 1

            evt_ray_dirs.append(vect)
            evt_ray_origins.append(origin)
            
    ray_origins.append(evt_ray_origins)
    ray_dirs.append(evt_ray_dirs)

    displays.append(ID_to_PE)


# store the events information
evts = {
    'evt_ids': evt_ids,
    'trk_origins': trk_origins,
    'trk_dirs': trk_dirs,
    'trk_lengths': trk_lengths,
    'ray_origins': ray_origins,
    'ray_dirs': ray_dirs,
    'PE': displays,
}

evts_df = pd.DataFrame(evts)
evts_df.to_hdf('test_evts.h5',key='data')
#-----------------------------

# store geom information
geom = {
    'positions': list(detector.all_points),
    'case': list(detector.ID_to_case.values()),
}

geom_df = pd.DataFrame(geom)
geom_df.to_hdf('test_geom.h5',key='data')
#-----------------------------

print('execution time: ', f"{time.time() - stime:.2f} seconds.")



