import numpy as np
import torch
from tqdm import tqdm 

SPEED_OF_LIGHT = 3E8

def backprojection(measurement : torch.Tensor,
                   det_locs : torch.Tensor, 
                   las_locs : torch.Tensor, 
                   x_l : torch.Tensor,
                   x_c : torch.Tensor,
                   voxel_params : dict,
                   t_res : float, 
                   device : str, 
                   num_batches : int,
                   t0: float,
                   r3_calib: bool = False
        ) -> torch.Tensor:

    """
    Compute backprojection from shadow transients.  

    Parameters:
    -----------
    measurement  : shadow transient measurement (num_pixels, num_bins)
    det_locs     : virtual detector locations (num_pixels, 3)
    las_locs     : virtual source locations (num_lasers, 3)
    x_l          : laser location (3, 1)
    x_c          : detector location (3, 1)
    voxel_params : dict containing voxel size and bounds of voxel grid
    t_res        : timing resolution / bin width in seconds
    device       : cpu/gpu/mps device
    num_batches  : number of batches to split voxels into for processing
    t0           : time corresponding to t=0 in seconds
    r3_calib     : toggle True if distance between camera and virtual detector
                    was already calibrated out in shadow_transients    

    Returns: 
    --------
    backprojected : voxel grid of backprojected intensity (num_x, num_y, num_z)

    """
    num_pixels = det_locs.shape[0]
    num_lasers = las_locs.shape[0]
    num_bins = measurement.shape[-1]

    measurement = measurement.unsqueeze(0)#.to(device)    
    las_locs = las_locs.to(device)
    det_locs = det_locs.to(device)
    x_l = x_l.to(device)
    x_c = x_c.to(device)

    # === Compute pathlength for each detector-laser pair === #
    binNum = compute_bin_numbers(det_locs, las_locs, x_l, x_c, t_res, t0, r3_calib) 
    binNum = torch.clip(binNum, 0, num_bins-1) # (num_pixels, num_lasers)

    # === Define voxel grid === #
    x_vals = voxel_params['x_vals']
    y_vals = voxel_params['y_vals']
    z_vals = voxel_params['z_vals']

    num_x = voxel_params['num_x']
    num_y = voxel_params['num_y']
    num_z = voxel_params['num_z']
    num_voxels = num_x * num_y * num_z

    spacing = voxel_params['spacing']

    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals) # (num_y, num_x, num_z)
    # x_vs_all = torch.stack([grid_x, grid_y, grid_z],0).reshape(3, -1) # (3, num_voxels)   
    x_vs_all = np.stack([grid_x.reshape(num_voxels, ), 
                         grid_y.reshape(num_voxels, ),
                         grid_z.reshape(num_voxels, )], axis=0) # (3, num_voxels)   
    
    x_vs_all = torch.Tensor(x_vs_all)

    # === Determine batch size === #
    batch_size = int(np.ceil(num_voxels / num_batches))

    # === Compute backprojection for batches of voxels === #
    backprojected = torch.zeros((num_voxels, )).to(device)
    for batch_idx in tqdm(range(num_batches)):
        # === Extract current batch of voxels === #
        idx1 = batch_size*batch_idx
        idx2 = min(batch_size*(batch_idx+1), num_voxels)
                   
        x_vs = x_vs_all[:, idx1:idx2].to(device) # (3, batch_size)
        cur_batch_size = x_vs.shape[-1]

        # === Compute ray-voxel intersection for voxels in batch and all detectors === #
        ray_voxel_dist = ray_voxel_distance(det_locs, las_locs, x_vs.T) # (cur_batch_size, num_pixels, num_lasers)
        hit_mask = ray_voxel_dist < np.cbrt(3/(4*np.pi)) * spacing # radius of sphere w/ same volume as cubic voxel        
        hit_idx = torch.nonzero(hit_mask, as_tuple=True)    

        # === Include voxel contribution to correct pixel/bin locations === #
        pixel_idxs = torch.arange(0, num_pixels, device=device)
        voxel_idxs = torch.arange(0, cur_batch_size, device=device)

        idx_3d = torch.stack([voxel_idxs[hit_idx[0]],
                              pixel_idxs[hit_idx[1]],
                              binNum[hit_idx[1], hit_idx[2]]], 0)

        spaceTime = torch.sparse_coo_tensor(idx_3d.to('cpu'),
                                            torch.ones(idx_3d.shape[1]),
                                            (cur_batch_size, num_pixels, num_bins))
        
        # === Compute backprojection (A^T @ y) === #
        backprojected[idx1:idx2] = torch.sum(measurement * spaceTime, dim=(1, 2)).to_dense()
                        
    backprojected = backprojected.cpu().numpy().reshape(num_y, num_x, num_z)
    backprojected = np.transpose(backprojected, (1, 0, 2))

    return backprojected

    
def compute_bin_numbers(det_locs : torch.Tensor,
                        las_locs : torch.Tensor,
                        x_l : torch.Tensor,
                        x_c : torch.Tensor,
                        t_res : float,
                        t0 : float,
                        r3_calib: bool = False
                    ) -> torch.Tensor:
    """
    Compute bin numbers for all possible light paths from 
    laser -> virtual source -> virtual detector -> detector.

    Parameters:
    -----------
    det_locs : virtual detector locations (num_pixels, 3)
    las_locs : virtual source locations (num_lasers, 3)
    x_l     : laser location (3, 1)
    x_c     : detector location (3, 1)
    t_res   : timing resolution in seconds
    t0      : time corresponding to t=0 in seconds
    r3_calib : toggle True to disregard distance from camera
                to virtual detector during ToF calculation

    Returns:
    --------
    nonzero_entries : (num_lasers, num_pixels)

    """

    x_l = x_l.reshape(1, 1, 3)
    x_c = x_c.reshape(1, 1, 3)
    det_locs = det_locs[:, None, :] # (num_pixels, 1, 3)
    las_locs = las_locs[None, :, :] # (1, num_lasers, 3)

    # === Compute distance between each light bounce === #

    r1 = torch.linalg.norm(x_l-las_locs, 
                           dim=-1, 
                           keepdim=True) # (1, num_lasers, 1)
    
    r2 = torch.linalg.norm(det_locs-las_locs, 
                           dim=-1, 
                           keepdim=True) # (num_pixels, num_lasers, 1)
    
    r3 = torch.linalg.norm(det_locs-x_c, 
                           dim=-1, 
                           keepdim=True) # (num_pixels, 1, 1)

    # == Compute bin number === #

    if r3_calib:
        pathLen = r1 + r2 # (num_pixels, num_lasers)
    else:
        pathLen = r1 + r2 + r3 # (num_pixels, num_lasers)

    tof = pathLen / SPEED_OF_LIGHT
    binNums = torch.round((tof - t0) / t_res).int()

    return binNums.squeeze() # (num_pixels, num_lasers)


def ray_voxel_distance(det_locs : torch.Tensor, 
                       las_locs : torch.Tensor, 
                       x_vs : torch.Tensor
                    ) -> torch.Tensor:
    """
    Compute distance of each voxel from closest point on ray 
    defined by each detector-laser pair. 

    Parameters:
    -----------
    det_locs : detector locations (num_pixels, 3)
    las_locs : laser locations (num_lasers, 3)
    x_vs     : voxel locations (num_voxels, 3)

    Returns:
    --------
    dist : closest distance from ray (num_voxels, num_pixels, num_lasers)

    """

    # === Add singleton dimension === #
    det_locs = det_locs[None, :, None, :] # (1, num_pixels, 1, 3)
    las_locs = las_locs[None, None, :, :] # (1, 1, num_lasers, 3)  
    x_vs = x_vs[:, None, None, :] # (num_voxels, 1, 1, 3)

    # === Compute ray directions from virtual laser -> virtual detector === #
    ray_dirs = normalize(det_locs-las_locs) # (1, num_pixels, num_lasers, 3)

    # === Compute perpendicular distance from ray === #
    voxel_laser = x_vs - las_locs # (num_voxels, 1, num_lasers, 3)
    parallel_comp = torch.sum(voxel_laser * ray_dirs, dim=-1, keepdim=True) * ray_dirs
    perp_comp = voxel_laser - parallel_comp # (num_voxels, num_pixels, num_lasers, 3)
    dist = torch.linalg.norm(perp_comp, dim=-1)

    return dist

    
def normalize(x: torch.Tensor, add_offset: bool = False):
    """
    Normalize last dimension of array to have norm 1.

    Parameters:
    -----------
    x  : array with last dimension the dimension to normalize (..., num_bins)

    Returns:
    --------
    y  : normalized array (..., num_bins)

    """

    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)

    if add_offset:
        x_norm = x_norm + 1e-7

    return x / x_norm    
