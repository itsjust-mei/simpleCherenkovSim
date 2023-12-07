# check_hits_cython.pyx
import numpy as np
cimport numpy as np

def check_hits_vectorized_per_track_cython(np.ndarray[np.float32_t, ndim=2] ray_origin,
                                            np.ndarray[np.float32_t, ndim=2] ray_direction,
                                            float sensor_radius,
                                            np.ndarray[np.float32_t, ndim=2] points):

    cdef np.ndarray[np.float32_t, ndim=3] vectors_to_points = points - ray_origin[:, np.newaxis, :]

    # Project all vectors onto the ray direction
    cdef np.ndarray[np.float32_t, ndim=2] dot_products_numerator = np.einsum('ijk,ik->ij', vectors_to_points, ray_direction)
    cdef np.ndarray[np.float32_t, ndim=1] dot_products_denominator = np.sum(ray_direction * ray_direction, axis=-1)

    # Calculate t_values
    cdef np.ndarray[np.float32_t, ndim=2] t_values = dot_products_numerator / dot_products_denominator[:, np.newaxis]

    # Calculate the points on the ray closest to the given points
    cdef np.ndarray[np.float32_t, ndim=3] closest_points_on_ray = ray_origin[:, np.newaxis, :] + t_values[:, :, np.newaxis] * ray_direction[:, np.newaxis, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    cdef np.ndarray[np.float32_t, ndim=2] distances = np.linalg.norm(points - closest_points_on_ray, axis=2)

    #cdef np.ndarray[bint, ndim=2] mask = t_values < 0
    distances[t_values < 0] = 999
    cdef np.ndarray[np.int32_t, ndim=1] indices = np.int32(np.argmin(distances, axis=1))


    cdef np.ndarray[np.int32_t, ndim=1] good_indices = indices[distances[range(len(indices)), indices] < sensor_radius]

    return good_indices
