from utils import load_configs, create_or_overwrite_directory
from data_loader import load_simulated_data, define_voxel_grid
from reconstruction import backprojection
from visualizations import save_max_projection
from time import time
import scipy.io
import os

configs = load_configs('config.yaml')

# === Import data === #
data_path = configs['data_path']
data = load_simulated_data(data_path, configs['spatial_ds'])

# === Define voxel grid === #
x_lims = configs['x_lims']
y_lims = configs['y_lims']
z_lims = configs['z_lims']

voxel_params = define_voxel_grid(x_lims, 
                                 y_lims, 
                                 z_lims,
                                 configs['spacing'])

# === Compute backprojection === #
time_start = time()
bp_rec = backprojection(measurement=data['shadow_transient'],
                        det_locs=data['det_locs'], 
                        las_locs=data['las_locs'], 
                        x_l=data['x_l'],
                        x_c=data['x_c'],
                        voxel_params=voxel_params,
                        t_res=data['t_res'], 
                        device=configs['device'], 
                        num_batches=configs['num_batches'],
                        t0=data['t0'],
                        r3_calib=configs['r3_calib'])

print(f'Elapsed time: {time()-time_start} s')

# === Save reconstructions === #
print("Saving outputs...")
save_dir = configs['save_dir']
create_or_overwrite_directory(save_dir)

# save max projection plots
save_max_projection(bp_rec, x_lims, y_lims, z_lims, save_dir)

# save voxel grid
save_dict = {"rho": bp_rec, "voxel_params": voxel_params}
save_file = os.path.join(save_dir, 'voxels.mat')
scipy.io.savemat(save_file, save_dict)

print("...done")
