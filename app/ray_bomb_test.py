import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from objects.geometry import *
from objects.ray import *
from objects.scene import *
from tools import *

# from geometry import *
# from ray import *

# -- example definitions (evenutally move to parameter file)
cyl_center        = np.array([0, 0, 0])
cyl_axis          = np.array([0, 0, 1]) # warning: visualization only works for [0,0,1]!
cyl_radius        = 6
cyl_height        = 10
cyl_barrel_grid   = [7,24]  # 4 cols and 10 rows
cyl_cap_rings     = [20,14,8,5,1] # 3 rings with 7,5,and 1 sensors.
cyl_sensor_radius = 0.5
ray_origin        = [0,0,0]
# ---------------------

# initialize your detector
scene    = Scene()

detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)
Nphot = 1000

counts_per_sensor = {i:0 for i in range(len(detector.all_points))}
for n in range(Nphot):

    ray_direction     = np.array([-1+2*np.random.uniform() for _ in range(3)])
    ray_direction     = ray_direction/np.linalg.norm(ray_direction)
    sensor_coordinates, dist, sensor_idx = detector.search_nearest_sensor_for_ray(ray_origin,ray_direction)

    if dist < cyl_sensor_radius:
        counts_per_sensor[sensor_idx] += 1

scene.add_photocounts(detector, counts_per_sensor)
scene.show(detector)