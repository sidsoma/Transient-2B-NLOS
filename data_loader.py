import torch
import numpy as np
import scipy.io

def load_simulated_data(data_path: str, spatial_ds: int):
    """
    Load simulated Mitsuba dataset.

    Parameters:
    -----------
    data_path : directory to data stored by .mat file
    spatial_ds : factor to downsample spatial resolution by
    r3_calib  : 

    Returns:
    --------
    Inputs to backprojection algorithm

    """
    data = scipy.io.loadmat(data_path)

    # === Camera and image parameters === #
    num_v, num_u = data['H'][0][0], data['W'][0][0] # number of pixels in y and x
    t0 = data['t_min'][0,0] / 3E8# in unit pathlength -> convert to time in seconds
    t_res = data['t_res'][0, 0] / 3E8 # in unit pathlength -> convert to time in seconds
    x_c = torch.Tensor([[0], [0], [0]]).to(torch.float) # camera location
    x_l = torch.Tensor([[0], [0], [0]]).to(torch.float) # laser location

    # === Import transient measurements === #
    shadow_transient = torch.Tensor(data['trans'])

    # === Import virtual detector locations (assumes two wall setup) === #
    img_wall_center = data['img_wall_center'][0]
    img_wall_size = data['img_wall_size'][0]

    y_locs = np.linspace(img_wall_center[1] - img_wall_size/2, 
                         img_wall_center[1] + img_wall_size/2, 
                         num_u)
    y_locs = np.flip(y_locs)
    z_locs = np.linspace(img_wall_center[2] - img_wall_size/2, 
                         img_wall_center[2] + img_wall_size/2, 
                         num_v)

    det_locs_x = img_wall_center[0] * np.ones((num_u, num_v))
    det_locs_z, det_locs_y = np.meshgrid(z_locs, y_locs)

    det_locs = np.stack((det_locs_x, det_locs_y, det_locs_z), axis=2)
    det_locs = torch.from_numpy(det_locs).to(torch.float)

    # === Import virtual source locations (assumes two wall setup) === #
    las_locs = torch.Tensor(data['l_coords'])

    # === Spatial downsample data === #
    print(f"Shadow transient shape before downsampling: {shadow_transient.shape}")
    shadow_transient = shadow_transient[::spatial_ds, ::spatial_ds, :]
    det_locs = det_locs[::spatial_ds, ::spatial_ds, :]
    print(f"Shadow transient shape after downsampling: {shadow_transient.shape}")

    # === Flatten histograms === #
    shadow_transient = torch.flatten(shadow_transient, 0, 1)
    det_locs = torch.flatten(det_locs, 0, 1)

    # === Store data === #
    bp_data = {'shadow_transient': shadow_transient, 'det_locs': det_locs, 'las_locs': las_locs, 
               'x_l': x_l, 'x_c': x_c, 't_res': t_res, 't0': t0}

    return bp_data

def define_voxel_grid(x_lims: list, y_lims: list, z_lims: list, spacing: float):
    """
    Parameters:
    -----------
    x_lims  : list containing min and max x bound (in meters)
    y_lims  : list containing min and max y bound (in meters)
    z_lims  : list containing min and max z bound (in meters)
    spacing : voxel size (in meters)

    Returns:
    --------
    voxel_params : config containing number of voxels and voxel locations
    
    """
    x_lims = np.array(x_lims)
    y_lims = np.array(y_lims)
    z_lims = np.array(z_lims)

    xmin = x_lims[0]; xmax = x_lims[1]
    num_x = 1+round((xmax-xmin)/spacing)
    x_vals = np.arange(xmin, xmax+spacing/2, spacing)

    ymin = y_lims[0]; ymax = y_lims[1]
    num_y = 1+round((ymax-ymin)/spacing)
    y_vals = np.arange(ymin, ymax+spacing/2, spacing)

    zmin = z_lims[0]; zmax = z_lims[1]
    num_z = 1+round((zmax-zmin)/spacing)
    z_vals = np.arange(zmin, zmax+spacing/2, spacing)

    voxel_params = {'x_vals': x_vals, 'y_vals': y_vals, 'z_vals': z_vals,
                    'num_x': num_x, 'num_y': num_y, 'num_z': num_z,
                    'spacing': spacing }
    
    return voxel_params

