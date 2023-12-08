import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
cyl_barrel_grid   = [14,48]  # 4 cols and 10 rows
cyl_cap_rings     = [40,34,30,24,20,16,8,5,1] # 3 rings with 7,5,and 1 sensors.
cyl_sensor_radius = 0.25

# ---------------------

# initialize your detector
scene    = Scene()

detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)

for i in range(10):

    track_origin        = [0,0,0]
    track_direction     = np.array([-1+2*np.random.uniform() for _ in range(3)])
    track_direction     = track_direction/np.linalg.norm(track_direction)

    scene.add_ray(track_origin,np.array(track_direction), L=4, c='r')
    track_direction = track_direction/np.linalg.norm(track_direction)

    A = detector.ray_cylinder_intersection(track_origin,np.array(track_direction))
    print(A)
    scene.add_point(A)

scene.show(detector)