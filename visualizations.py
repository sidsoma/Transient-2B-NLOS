import matplotlib.pyplot as plt
import numpy as np
import os


def save_max_projection(rho: np.array, 
                        x_lims: np.array,
                        y_lims: np.array,
                        z_lims: np.array,
                        save_dir: str
                    ):
    """
    Plot max projection of voxel reconstructions.

    Parameters:
    -----------
    rho      : np.array containing voxel opacity (num_x, num_y, num_z)
    x_lims   : list containing min and max x bound (in meters)
    y_lims   : list containing min and max y bound (in meters)
    z_lims   : list containing min and max z bound (in meters)
    save_dir : folder to save plot to 
    
    """

    plot_rho = np.copy(rho[10:-10, :, :])

    x_lims = [-0.5, 0.5]
    y_lims = [-0.5, 0.5]
    z_lims = [1, 2]

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title('Front View')
    plt.imshow(np.max(plot_rho, 0).T, extent=[z_lims[1], z_lims[0], y_lims[0], y_lims[1]], cmap='hot')
    plt.xlabel('z')
    plt.ylabel('y')

    plt.subplot(1, 3, 2)
    plt.title('Top View')
    plt.imshow(np.max(plot_rho, 1).T, extent=[x_lims[0], x_lims[1], z_lims[0], z_lims[1]], cmap='hot')
    plt.xlabel('x')
    plt.ylabel('z')


    plt.subplot(1, 3, 3)
    plt.title('Side View')
    plt.imshow(np.max(plot_rho, 2).T, extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]], cmap='hot')
    plt.xlabel('x')
    plt.ylabel('y')


    plt.savefig(os.path.join(save_dir, 'reconstructions.png'))

