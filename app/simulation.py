import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from objects.geometry import *
from objects.ray import *
from objects.segment import *
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
ray_origin        = [0,0,2]#np.array([-cyl_radius+2*np.random.uniform(cyl_radius), -cyl_radius+2*np.random.uniform(cyl_radius), -cyl_height/2+np.random.uniform(cyl_height)])
ray_direction     = np.array([np.random.uniform()/np.sqrt(2) for _ in range(3)])
ray_direction     = ray_direction/np.linalg.norm(ray_direction)
# ---------------------

# initialize your detector
scene    = Scene()
detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)
ray      = Ray(ray_origin,ray_direction)

sensor_coordinates, dist, sensor_idx = detector.search_nearest_sensor_for_ray(ray_origin,ray_direction)

ray_length = 1.*point_to_point_dist(ray_origin,sensor_coordinates)

print(ray_length)

scene.add_photosensors(detector)
color = 'r' if dist < cyl_sensor_radius else 'b'

scene.add_point(sensor_coordinates, c=color)
scene.add_ray(ray_origin,ray_direction, L=ray_length, c=color)

scene.show(detector)