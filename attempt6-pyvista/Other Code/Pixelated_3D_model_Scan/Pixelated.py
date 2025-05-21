import trimesh
import numpy as np
import matplotlib.pyplot as plt

def simulate_sd_oct(obj_file, axial_res=7, lateral_res=8, batch_size=100000):
    """
    Simulate SD-OCT imaging on a 3D model in OBJ format by voxelizing the mesh.
    
    Parameters:
      obj_file (str): Path to the OBJ file.
      axial_res (float): Axial resolution in mm.
      lateral_res (float): Lateral resolution in mm.
      batch_size (int): Number of points to process per batch.
    
    Returns:
      voxels (numpy.ndarray): 3D boolean array representing the voxelized model.
      grid (tuple): Tuple of (x, y, z) arrays representing the voxel grid coordinates.
    """
    # Load the mesh from the OBJ file
    mesh = trimesh.load(obj_file)
    
    # Get the mesh bounds (min and max coordinates)
    bounds = mesh.bounds  # shape (2, 3): [min, max]
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # Create coordinate vectors for the grid based on the resolutions.
    x = np.arange(x_min, x_max, lateral_res)
    y = np.arange(y_min, y_max, lateral_res)
    z = np.arange(z_min, z_max, axial_res)
    
    # Create a grid of points corresponding to the center of each voxel.
    X, Y, Z = np.meshgrid(x + lateral_res/2,
                          y + lateral_res/2,
                          z + axial_res/2,
                          indexing='ij')
    
    # Flatten the grid to a list of points
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Instead of checking all points at once, process in batches.
    inside = np.empty(points.shape[0], dtype=bool)
    num_points = points.shape[0]
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        inside[start:end] = mesh.contains(points[start:end])
    
    # Reshape the result into the 3D voxel grid
    voxels = inside.reshape(X.shape)
    
    return voxels, (x, y, z)

def visualize_voxels(voxels, voxel_aspect=(8, 8, 7)):
    """
    Visualize a 3D voxel grid using matplotlib's voxel plotting.
    
    Parameters:
      voxels (numpy.ndarray): 3D boolean array representing the voxelized model.
      voxel_aspect (tuple): Relative aspect ratio of the voxels (lateral, lateral, axial).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the voxels with edge color to delineate voxel boundaries
    ax.voxels(voxels, edgecolor='k')
    
    # Set the aspect ratio based on the voxel sizes
    ax.set_box_aspect(voxel_aspect)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Simulated SD-OCT Voxel Image")
    plt.show()

if __name__ == '__main__':
    # Path to your OBJ file
    obj_file = '/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj'
    voxels, grid = simulate_sd_oct(obj_file)
    
    # Optionally, save the voxel grid for further analysis.
    np.save('voxel_image.npy', voxels)
    
    # Visualize the voxelized model.
    visualize_voxels(voxels, voxel_aspect=(8, 8, 7))
