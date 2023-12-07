import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from objects.geometry import *
from objects.ray import *
from objects.scene import *
from tools import *

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

detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)
Nphot = 1000000

ID_to_PE = {i:0 for i in range(len(detector.all_points))}
for n in range(Nphot):

    ray_direction = generate_isotropic_random_vector()
    ray_direction     = ray_direction/np.linalg.norm(ray_direction)
    # ray_direction     = np.array([-1+2*np.random.uniform() for _ in range(3)])
    
    sensor_coordinates, dist, sensor_idx = detector.search_nearest_sensor_for_ray(ray_origin,ray_direction)

    if dist < cyl_sensor_radius:
        ID_to_PE[sensor_idx] += 1

# uncomment this to have 3D display
# scene    = Scene()
# scene.add_photocounts(detector, ID_to_PE)
# scene.show(detector)

# uncomment this to have 2D display
# show_2D_display(detector.ID_to_position, ID_to_PE, detector.ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height, file_name='isotropic_dist.pdf')


# grouping light yield per distance
R_to_PE    = {}
for ID in list(detector.ID_to_position.keys()):
    R = f"{point_to_point_dist([0,0,0],detector.ID_to_position[ID]):.2f}"
    R_to_PE[R] =  []

for ID in list(detector.ID_to_position.keys()):
    R = f"{point_to_point_dist([0,0,0],detector.ID_to_position[ID]):.2f}"
    R_to_PE[R].append(ID_to_PE[ID])

x = []
y = []
y_err = []
for key,value in R_to_PE.items():
    x.append(key)
    y.append(np.mean(value))
    y_err.append(np.std(value))

print(x)
x = [float(i) for i in x]
x = np.array(x)

plt.errorbar(x, y*(x**2), y_err, fmt='o', color='k')

plt.gca().set_ylim(0., 1.5*max(y*(x**2)))

plt.gca().set_ylabel('<PE> * r^2')
plt.gca().set_xlabel('<PE>')

plt.show()




