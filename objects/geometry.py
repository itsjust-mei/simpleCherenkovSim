import numpy as np

class Cylinder:
    def __init__(self, center, axis, radius, height, barrel_grid, cap_rings, cyl_sensor_radius):

        self.C = center
        self.A = axis
        self.r = radius
        self.H = height 
        self.S_radius = cyl_sensor_radius

        self.place_photosensors(barrel_grid,cap_rings)

    def ray_cylinder_intersection(self, ray_origin, ray_direction):
        # Ray: P = O + tD, where P is a point on the ray, O is the ray origin, D is the ray direction, and t is a parameter.
        # Cylinder: (P - C - ((P - C) dot A)A)^2 = r^2, where C is the cylinder center, A is the cylinder axis, and r is the radius.

        # Define the variables
        O = np.array(ray_origin)
        D = np.array(ray_direction)
        C = self.C
        A = self.A
        r = self.r
        H = self.H

        # Calculate coefficients for the quadratic equation
        a = np.dot(D - np.dot(D, A) * A, D - np.dot(D, A) * A)
        b = 2 * np.dot(D - np.dot(D, A) * A, O - C - np.dot(O - C, A) * A)
        c = np.dot(O - C - np.dot(O - C, A) * A, O - C - np.dot(O - C, A) * A) - r**2

        # Solve the quadratic equation
        discriminant = b**2 - 4 * a * c

        if a>=0:
            # Calculate the solutions for t (parameter along the ray)
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Check if the intersection points are in the positive direction of the ray
            intersection_point1 = O + t1 * D if t1 >= 0 and 0 <= (O + t1 * D - C).dot(A) <= H/2 else None
            intersection_point2 = O + t2 * D if t2 >= 0 and 0 <= (O + t2 * D - C).dot(A) <= H/2 else None

            if intersection_point1 is not None or intersection_point2 is not None:
                return intersection_point1, intersection_point2, None, None

        # If no intersection with the barrel, check for intersection with the caps
        t_cap1 = (H/2 - (O - C).dot(A)) / D.dot(A) if D.dot(A) != 0 else np.inf
        t_cap2 = -((O - C).dot(A)) / D.dot(A) if D.dot(A) != 0 else np.inf

        cap_intersection1 = O + abs(t_cap1) * D
        cap_intersection2 = O + abs(t_cap2) * D

        if t_cap1 == 0:
            cap_intersection1 = None

        if t_cap2 == 0:
            cap_intersection2 = None
        
        return None, None, cap_intersection1, cap_intersection2

    def distance_ray_point(self, ray_origin, ray_direction, point):
        # Calculate the vector from ray origin to the point
        vector_to_point = np.array(point) - np.array(ray_origin)
        
        # Project the vector onto the ray direction
        t = np.dot(vector_to_point, ray_direction) / np.dot(ray_direction, ray_direction)
        
        # Calculate the point on the ray closest to the given point
        closest_point_on_ray = np.array(ray_origin) + t * np.array(ray_direction)
        
        # Calculate the Euclidean distance between the two points
        distance = np.linalg.norm(np.array(point) - closest_point_on_ray)
        
        return distance

    def check_direction_of_points(self, ray_origin, ray_direction, point1, point2, idx1, idx2):

        # Calculate distances from the ray origin to each point along the ray direction
        distance_to_point1 = np.dot(np.array(point1) - np.array(ray_origin), ray_direction)
        distance_to_point2 = np.dot(np.array(point2) - np.array(ray_origin), ray_direction)

        # Determine which point is in the direction of the ray
        if distance_to_point1 < distance_to_point2:
            return idx2
        elif distance_to_point2 < distance_to_point1:
            return idx1
        else:
            return None

    def search_nearest_sensor_for_ray(self, ray_origin, ray_direction):
        distances  = [self.distance_ray_point(ray_origin,ray_direction,p) for p in self.all_points]
        indices    = np.argsort(distances)
        found_idx  = self.check_direction_of_points(ray_origin, ray_direction, self.all_points[indices[0]], self.all_points[indices[1]], indices[0], indices[1])
        return self.all_points[found_idx], distances[found_idx], found_idx

    def place_photosensors(self, barrel_grid, cap_rings):

        # barrel ----
        b_rows = barrel_grid[0]
        b_cols = barrel_grid[1]

        theta = np.linspace(0, 2*np.pi, b_cols, endpoint=False)  # Generate N angles from 0 to 2pi
        x = self.r * np.cos(theta) + self.C[0]
        y = self.r * np.sin(theta) + self.C[1]
        z = [(i+1)*self.H/(b_rows+1)-self.H/2 + self.C[2] for i in range(b_rows)]

        barr_points = np.array([[x[j],y[j],z[i]] for i in range(b_rows) for j in range(b_cols)])
        self.barr_points = barr_points

        del x,y,z,theta # ensure no values are passed to the caps.
        # -----------

        # caps ----
        Nrings = len(cap_rings)

        tcap_points = []
        bcap_points = []
        for i_ring, N_sensors_in_ring in enumerate(cap_rings):
            theta = np.linspace(0, 2*np.pi, N_sensors_in_ring, endpoint=False)  # Generate N angles from 0 to 2pi
            x = self.r*((Nrings-(i_ring+1))/Nrings)* np.cos(theta) + self.C[0]
            y = self.r*((Nrings-(i_ring+1))/Nrings)* np.sin(theta) + self.C[1]
            top_z = [ self.H/2 + self.C[2] for i in range(N_sensors_in_ring)]
            bot_z = [-self.H/2 + self.C[2] for i in range(N_sensors_in_ring)]

            for i_sensor in range(N_sensors_in_ring):
                tcap_points.append([x[i_sensor],y[i_sensor],top_z[i_sensor]])
                bcap_points.append([x[i_sensor],y[i_sensor],bot_z[i_sensor]])

        self.tcap_points = np.array(tcap_points)
        self.bcap_points = np.array(bcap_points)

        self.all_points = np.concatenate([self.barr_points, self.tcap_points, self.bcap_points],axis=0)
        # -----------


