import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

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
ID_to_PE = {i:0 for i in range(len(detector.all_points))}

stime = time.time()

for i in range(5):
    track_origin        = [0,1,2.5]
    track_direction     = np.array([-1+2*np.random.uniform() for _ in range(3)])
    track_direction     = track_direction/np.linalg.norm(track_direction)

    # in this example we have only one segment for all the track
    track = Segment(track_origin, track_direction, length=cyl_radius)
    track.gen_ch_rays(np.radians(40), num_samples=1000)

    Nrays = len(track.ray_vectors)

    ray_vectors_x = track.ray_vectors[:,0]
    ray_vectors_y = track.ray_vectors[:,1]
    ray_vectors_z = track.ray_vectors[:,2]
    vect = [ray_vectors_x,ray_vectors_y,ray_vectors_z]

    for n in range(Nrays):

        vect = [ray_vectors_x[n],ray_vectors_y[n],ray_vectors_z[n]]
        vect = vect/np.linalg.norm(vect)

        origin = (0.5+2*np.random.uniform())*track_direction*(n/Nrays)

        sensor_coordinates, dist, sensor_idx = detector.search_nearest_sensor_for_ray(origin,vect)

        if dist < cyl_sensor_radius:
            ID_to_PE[sensor_idx] += 1


print('execution time: ', f"{time.time()-stime:.2f}")
show_2D_display(detector.ID_to_position, ID_to_PE, detector.ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height, file_name='evt_example.pdf')






