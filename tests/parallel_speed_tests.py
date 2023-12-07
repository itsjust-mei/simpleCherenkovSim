import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

from objects.geometry import *
from objects.ray import *
from objects.segment import *
from objects.scene import *
from tools import *

from check_hits_cython import check_hits_vectorized_per_track_cython

import numba
import dask.array as da
import dask

### ---- all functions we want to test...

# for a given ray we vectorize calculation over all photosensors
def check_hits_vectorized_per_ray(ray_origin, ray_direction, sensor_radius, points):
    # Calculate vectors from ray origin to all points
    vectors_to_points = points - ray_origin

    # Project all vectors onto the ray direction
    t_values = np.dot(vectors_to_points, ray_direction) / np.dot(ray_direction, ray_direction)

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin + t_values[:, np.newaxis] * ray_direction

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = np.linalg.norm(points - closest_points_on_ray, axis=1)

    mask = t_values < 0
    distances[mask] = 999
    idx = np.argmin(distances)
    
    if distances[idx]<sensor_radius:
        return idx
    else:
        return None

def check_hits_vectorized_per_track(ray_origin, ray_direction, sensor_radius, points):
    # Calculate vectors from ray origin to all points
    vectors_to_points = points - ray_origin[:, np.newaxis, :]

    # Project all vectors onto the ray direction using einsum
    dot_products_numerator = np.einsum('ijk,ik->ij', vectors_to_points, ray_direction)
    dot_products_denominator = np.sum(ray_direction * ray_direction, axis=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, np.newaxis]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin[:, np.newaxis, :] + t_values[:, :, np.newaxis] * ray_direction[:, np.newaxis, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = np.linalg.norm(points - closest_points_on_ray, axis=2)

    mask = t_values < 0
    distances[mask] = 999
    indices = np.argmin(distances, axis=1)
    good_indices = indices[distances[range(len(indices)), indices] < sensor_radius]

    return good_indices


def check_hits_vectorized_per_track_dask(ray_origin, ray_direction, sensor_radius, points):
    # Convert input arrays to Dask arrays
    ray_origin_da = da.from_array(ray_origin, chunks=ray_origin.shape)
    ray_direction_da = da.from_array(ray_direction, chunks=ray_direction.shape)
    points_da = da.from_array(points, chunks=points.shape)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_da - ray_origin_da[:, np.newaxis, :]

    # Project all vectors onto the ray direction using einsum
    dot_products_numerator = da.einsum('ijk,ik->ij', vectors_to_points, ray_direction_da)
    dot_products_denominator = da.sum(ray_direction_da * ray_direction_da, axis=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, np.newaxis]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_da[:, np.newaxis, :] + t_values[:, :, np.newaxis] * ray_direction_da[:, np.newaxis, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = da.linalg.norm(points_da - closest_points_on_ray, axis=2)

    # Convert sensor_radius to Dask array
    sensor_radius_da = da.from_array(sensor_radius)

    # Create a mask and apply it to distances
    mask = t_values < 0
    distances = da.where(mask, 999, distances)

    # Find the indices of the minimum distances using NumPy
    indices_np = np.argmin(distances.compute(), axis=1)

    # Delayed indexing operation using dask.delayed
    @dask.delayed
    def delayed_indexing(distances, indices, sensor_radius):
        result = []
        for d, i in zip(distances, indices):
            if d[i] < sensor_radius:
                result.append(i)
        return result

    result = delayed_indexing(distances, indices_np, sensor_radius_da)
    
    # Compute the result
    return da.compute(result)[0]

import torch

def check_hits_vectorized_per_track_torch(ray_origin, ray_direction, sensor_radius, points):

    # Convert NumPy arrays to PyTorch tensors
    ray_origin_torch = torch.tensor(ray_origin, dtype=torch.float32)
    ray_direction_torch = torch.tensor(ray_direction, dtype=torch.float32)
    points_torch = torch.tensor(points, dtype=torch.float32)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_torch - ray_origin_torch[:, None, :]

    # Project all vectors onto the ray direction using einsum
    dot_products_numerator = torch.einsum('ijk,ik->ij', vectors_to_points, ray_direction_torch)
    dot_products_denominator = torch.sum(ray_direction_torch * ray_direction_torch, dim=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, None]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_torch[:, None, :] + t_values[:, :, None] * ray_direction_torch[:, None, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = torch.norm(points_torch - closest_points_on_ray, dim=2)

    # Apply the mask
    mask = t_values < 0
    distances[mask] = 999.0

    # Find the indices of the minimum distances
    indices = torch.argmin(distances, dim=1)

    # Get the good indices based on sensor_radius
    good_indices = indices[distances[torch.arange(indices.size(0)), indices] < sensor_radius]

    return good_indices.numpy()


def check_hits_vectorized_per_track_torch_gpu(ray_origin, ray_direction, sensor_radius, points):

    device = torch.device("mps")

    # Convert NumPy arrays to PyTorch tensors and move to "mps" device
    ray_origin_torch = torch.tensor(ray_origin, dtype=torch.float32, device=device)
    ray_direction_torch = torch.tensor(ray_direction, dtype=torch.float32, device=device)
    points_torch = torch.tensor(points, dtype=torch.float32, device=device)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_torch - ray_origin_torch[:, None, :]

    # Project all vectors onto the ray direction using einsum
    dot_products_numerator = torch.einsum('ijk,ik->ij', vectors_to_points, ray_direction_torch)
    dot_products_denominator = torch.sum(ray_direction_torch * ray_direction_torch, dim=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, None]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_torch[:, None, :] + t_values[:, :, None] * ray_direction_torch[:, None, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = torch.norm(points_torch - closest_points_on_ray, dim=2)

    # Apply the mask
    mask = t_values < 0
    distances = torch.where(mask, torch.tensor(999.0, device=device), distances)

    # Find the indices of the minimum distances
    indices = torch.argmin(distances, dim=1)

    # Get the good indices based on sensor_radius
    good_indices = indices[distances[torch.arange(indices.size(0)), indices] < sensor_radius]

    return good_indices.cpu().numpy()


# we use numba. 
@numba.jit(nopython=True)
def check_hits_numba(ray_origin, ray_direction, sensor_radius, points):
    num_rays = ray_origin.shape[0]
    num_points = points.shape[0]

    # Initialize an array to store vectors_to_points
    vectors_to_points = np.empty((num_rays, num_points, 3), dtype=np.float32)

    # Calculate vectors from ray origin to all points using basic loops
    for i in range(num_rays):
        for j in range(num_points):
            for z in range(3):
                vectors_to_points[i, j, z] = points[j, z] - ray_origin[i, z]

    # Initialize arrays to store dot_products_numerator, dot_products_denominator, and closest_points_on_ray
    dot_products_numerator = np.empty((num_rays, num_points), dtype=np.float32)
    dot_products_denominator = np.empty((num_rays,), dtype=np.float32)
    closest_points_on_ray = np.empty((num_rays, num_points, 3), dtype=np.float32)

    # Calculate dot_products_numerator and dot_products_denominator using basic loops
    for i in range(num_rays):
        for j in range(num_points):
            dot_products_numerator[i, j] = 0.0
            for z in range(3):
                dot_products_numerator[i, j] += vectors_to_points[i, j, z] * ray_direction[i, z]

        dot_products_denominator[i] = 0.0
        for z in range(3):
            dot_products_denominator[i] += ray_direction[i, z] * ray_direction[i, z]

    # Initialize arrays to store distances and mask
    distances = np.empty((num_rays, num_points), dtype=np.float32)
    mask = np.empty((num_rays, num_points), dtype=np.bool_)

    # Calculate distances using basic loops
    for i in range(num_rays):
        for j in range(num_points):
            t_values = dot_products_numerator[i, j] / dot_products_denominator[i]
            closest_points_on_ray[i, j] = ray_origin[i] + t_values * ray_direction[i]

            distances[i, j] = 0.0
            for z in range(3):
                distances[i, j] += (points[j, z] - closest_points_on_ray[i, j, z]) ** 2
            distances[i, j] = np.sqrt(distances[i, j])

            mask[i, j] = t_values < 0

    # Apply the mask to set distances to a large value for invalid t_values using basic loops
    for i in range(num_rays):
        for j in range(num_points):
            if mask[i, j]:
                distances[i, j] = 999

    # Find the indices of points within the sensor radius using basic loops
    indices = np.empty((num_rays,), dtype=np.int64)
    for i in range(num_rays):
        min_distance_index = 0
        min_distance = distances[i, 0]
        for j in range(1, num_points):
            if distances[i, j] < min_distance:
                min_distance = distances[i, j]
                min_distance_index = j
        indices[i] = min_distance_index

    # Filter indices based on sensor radius using basic loops
    good_indices = []
    for i in range(len(indices)):
        if distances[i, indices[i]] < sensor_radius:
            good_indices.append(indices[i])

    return good_indices

### ------------------------------------------------------------------------

# -- geom definitions
cyl_center        = np.array([0, 0, 0])
cyl_axis          = np.array([0, 0, 1]) # warning: visualization only works for [0,0,1]!
cyl_radius        = 6
cyl_height        = 10
cyl_barrel_grid   = [14,48]  # 14 cols and 48 rows
cyl_cap_rings     = [40,34,30,24,20,16,10,5,1] # 9 concentric rings with a number of sensors specified by the array
cyl_sensor_radius = 0.25
# ---------------------
detector = Cylinder(cyl_center, cyl_axis, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius)


# -- the speed tests

Nphot = 100000
Ntrk  = 1

print(torch.cuda.is_available())

# PURE NUMPY VECTORIZING ONLY DIST CALC
ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)
        idx = check_hits_vectorized_per_ray(np.array(ray_origin, dtype=np.float32),np.array(ray_direction, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    
        if idx is not None:
            ID_to_PE[idx] +=1

print('1) Ray vect. np exec. time: ', f"{time.perf_counter()-stime:.2f} s.")


# PURE NUMPY ALL VECT
ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    vects = []
    origs = []
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)

        vects.append(ray_direction)
        origs.append(ray_origin)

    # USING VECTORIZATION WITH NUMPY  
    good_indices = check_hits_vectorized_per_track(np.array(origs, dtype=np.float32),np.array(vects, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    idx, cts = np.unique(good_indices, return_counts=True)
    ID_to_PE[idx] = cts
print('2) Track vect. np exec. time: ', f"{time.perf_counter()-stime:.2f} s.")

ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    vects = []
    origs = []
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)

        vects.append(ray_direction)
        origs.append(ray_origin)

    # USING NUMBA
    good_indices = check_hits_numba(np.array(origs, dtype=np.float32),np.array(vects, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    idx, cts = np.unique(good_indices, return_counts=True)
    ID_to_PE[idx] = cts
print('3) Numba exec. time: ', f"{time.perf_counter()-stime:.2f} s.")

# PURE NUMPY ALL VECT
ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    vects = []
    origs = []
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)

        vects.append(ray_direction)
        origs.append(ray_origin)

    # USING VECTORIZATION WITH NUMPY  
    good_indices = check_hits_vectorized_per_track_dask(np.array(origs, dtype=np.float32),np.array(vects, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    idx, cts = np.unique(good_indices, return_counts=True)
    ID_to_PE[idx] = cts
print('4) Dask exec. time: ', f"{time.perf_counter()-stime:.2f} s.")


### CYTHON
ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    vects = []
    origs = []
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)

        vects.append(ray_direction)
        origs.append(ray_origin)

    # USING CYTHON
    good_indices = check_hits_vectorized_per_track_cython(np.array(origs, dtype=np.float32),np.array(vects, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    idx, cts = np.unique(good_indices, return_counts=True)
    ID_to_PE[idx] = cts
print('5) Cython exec. time: ', f"{time.perf_counter()-stime:.2f} s.")

### GPU
ID_to_PE = np.zeros(len(detector.all_points))
stime = time.perf_counter()
for i in range(Ntrk):
    vects = []
    origs = []
    ray_origin        = [0,0,0]
    for n in range(Nphot):

        ray_direction = generate_isotropic_random_vector()
        ray_direction = ray_direction/np.linalg.norm(ray_direction)

        vects.append(ray_direction)
        origs.append(ray_origin)

    # USING TORCH
    good_indices = check_hits_vectorized_per_track_torch_gpu(np.array(origs, dtype=np.float32),np.array(vects, dtype=np.float32),cyl_sensor_radius, np.array(detector.all_points,dtype=np.float32))
    idx, cts = np.unique(good_indices, return_counts=True)
    ID_to_PE[idx] = cts
print('6) Torch CPU exec. time: ', f"{time.perf_counter()-stime:.2f} s.")
