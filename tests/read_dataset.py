import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import pandas as pd

from objects.geometry import *
from objects.ray import *
from objects.segment import *
from objects.scene import *
from tools import *

df = pd.read_hdf('test_evts.h5')  

print(df)

#print(np.shape(df.ray_origins))