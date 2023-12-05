import numpy as np
from mpl_toolkits.mplot3d import art3d


def point_to_point_dist(p1,p2):
    return np.sqrt(np.sum([(p1[i]-p2[i])**2 for i in range(3)]))
