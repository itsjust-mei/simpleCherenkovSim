import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize


def create_color_gradient(max_cnts, colormap='viridis'):
    """Define the color scale in the 2D event display"""

    # Define the colormap and normalization
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_cnts)

    # Create a scalar mappable
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    return scalar_mappable

def show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height, file_name=None):
    """Do the 2D event display"""
    max_PE = np.max(list(ID_to_PE.values()))
    color_gradient = create_color_gradient(max_PE)

    fig, ax = plt.subplots(figsize=(8,8),facecolor='black')

    for ID in list(ID_to_position.keys()):
        pos   = ID_to_position[ID]
        PE    = ID_to_PE[ID]
        case  = ID_to_case[ID]

        caps_offset = 0.05

        #barrel
        if case ==0:
            theta = np.arctan(pos[1]/pos[0]) if pos[0] != 0 else np.pi/2
            theta += np.pi/2
            if pos[0]>0:
                theta += np.pi
            theta /=2
            z = pos[2]/cyl_height

            ax.add_patch(plt.Circle((theta, z), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

        elif case ==1:
            ax.add_patch(plt.Circle((pos[0]/cyl_height+np.pi/2, 1+caps_offset+pos[1]/cyl_height), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

        elif case ==2:
            ax.add_patch(plt.Circle((pos[0]/cyl_height+np.pi/2, -1-caps_offset-pos[1]/cyl_height), cyl_sensor_radius/cyl_height, color=color_gradient.to_rgba(PE)))

    margin = 0.05

    ax.set_facecolor("black")

    #hide x-axis
    ax.get_xaxis().set_visible(False)
    #hide y-axis 
    ax.get_yaxis().set_visible(False)
    plt.axis('equal')
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()

# Replace this with your actual filename
filename = 'datasets/sim_mode_0_dataset_0_events.h5'
geom_df  = pd.read_hdf('datasets/sim_mode_0_dataset_geom.h5')

# Open the HDF5 file in read mode
with h5py.File(filename, 'r') as f:

    # Access datasets
    h5_evt_ids = f['evt_id']
    h5_evt_pos = f['positions']
    h5_evt_hit_idx = f['event_hits_index']
    h5_evt_hit_IDs = f['hit_pmt']
    h5_evt_hit_Qs = f['hit_charge']
    h5_evt_hit_Ts = f['hit_time']

    # Access data
    evt_ids = h5_evt_ids[:]
    evt_pos = h5_evt_pos[:]
    evt_hit_idx = h5_evt_hit_idx[:]
    evt_hit_IDs = h5_evt_hit_IDs[:]
    evt_hit_Qs = h5_evt_hit_Qs[:]
    evt_hit_Ts = h5_evt_hit_Ts[:]

# Now you can use the retrieved data as needed
# For example, printing the first 5 elements of each dataset
# print("Event IDs:", evt_ids[:])
# print("Event Positions:", evt_pos[:])
# print("Event Hit Indices:", evt_hit_idx[:])
# print("Event Hit IDs:", evt_hit_IDs[:])
# print("Event Hit Charges:", evt_hit_Qs[:])
# print("Event Hit Times:", evt_hit_Ts[:])

# -- geom definitions --- this are just some numbers that are thought to fill a pseudo-realistic cylinder 
cyl_center        = np.array([0, 0, 0])
cyl_axis          = np.array([0, 0, 1]) # warning: visualization only works for [0,0,1]!
cyl_radius        = 6
cyl_height        = 10
cyl_barrel_grid   = [14,48]  # 14 cols and 48 rows
cyl_cap_rings     = [40,34,30,24,20,16,10,5,1] # 9 concentric rings with a number of sensors specified by the array
cyl_sensor_radius = 0.25
# ---------------------

IDs = None

evt_ID = 1
if evt_ID == 0:
    IDs = evt_hit_IDs[0:evt_hit_idx[0]]
    Qs  = evt_hit_Qs[0:evt_hit_idx[0]]
    Ts  = evt_hit_Ts[0:evt_hit_idx[0]]
else:
    IDs = evt_hit_IDs[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]
    Qs  = evt_hit_Qs[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]
    Ts  = evt_hit_Ts[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]


ID_to_PE = np.zeros(len(geom_df.positions))
ID_to_PE[IDs] = Qs

ID_to_position = {i:x for i,x in enumerate(geom_df.positions)}
ID_to_case = {i:x for i,x in enumerate(geom_df.case)}
ID_to_PE = {i:x for i,x in enumerate(ID_to_PE)}

show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height)#, file_name='evt_example.pdf')







