import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import pandas as pd

from objects.geometry import *
from objects.ray import *
from objects.segment import *
from objects.scene import *
from tools import *

# store the events information
evts = {
    'evt_ids': evt_ids,
    'all_origins': all_origins,
    'all_dirs': all_dirs,
    'all_lengths': all_lengths,
    'PE': displays,
}

evts_df = pd.DataFrame(evts)
evts_df.to_hdf('test_evts.h5',key='data')
#-----------------------------

# store geom information
geom = {
    'positions': list(detector.all_points),
    'case': list(detector.ID_to_case.values()),
}

geom_df = pd.DataFrame(geom)
geom_df.to_hdf('test_geom.h5',key='data')
#-----------------------------

print('execution time: ', f"{time.time() - stime:.2f} seconds.")