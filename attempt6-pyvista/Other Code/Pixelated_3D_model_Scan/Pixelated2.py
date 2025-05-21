import trimesh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import psutil

def simulate_sd_oct(obj_file, axial_res=7, lateral_res=8, batch_size=100000, max_voxels=50000000):
    """
    Simulate SD-OCT imaging on a 3D model in OBJ format by voxelizing the mesh.
    
    Parameters:
      obj_file (str): Path to the OBJ file.
      axial_res (float): Axial resolution in mm.
      lateral_res (float): Lateral resolution in mm.
      batch_size (int): Number of points to process per batch.
      max_voxels (int): Maximum number of voxels to process to prevent memory issues.
    
    Returns:
      voxels (numpy.ndarray): 3D boolean array representing the voxelized model.
      grid (tuple): Tuple of (x, y, z) arrays representing the voxel grid coordinates.
    """
    print(f"Loading mesh from {obj_file}...")
    # Load the mesh from the OBJ file
    mesh = trimesh.load(obj_file)
    
    # Get the mesh bounds (min and max coordinates)
    bounds = mesh.bounds  # shape (2, 3): [min, max]
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    print(f"Mesh bounds: {bounds}")
    
    # Calculate the total number of voxels that would be created
    x_points = int(np.ceil((x_max - x_min) / lateral_res))
    y_points = int(np.ceil((y_max - y_min) / lateral_res))
    z_points = int(np.ceil((z_max - z_min) / axial_res))
    total_voxels = x_points * y_points * z_points
    
    print(f"Grid dimensions: {x_points} x {y_points} x {z_points} = {total_voxels} voxels")
    
    # Check if the number of voxels is too large
    if total_voxels > max_voxels:
        print(f"Warning: Large voxel count ({total_voxels}) exceeds maximum ({max_voxels})")
        print("Adjusting resolution to fit memory constraints...")
        
        # Adjust resolution to fit within max_voxels
        scale_factor = (total_voxels / max_voxels) ** (1/3)
        lateral_res *= scale_factor
        axial_res *= scale_factor
        
        # Recalculate grid dimensions
        x_points = int(np.ceil((x_max - x_min) / lateral_res))
        y_points = int(np.ceil((y_max - y_min) / lateral_res))
        z_points = int(np.ceil((z_max - z_min) / axial_res))
        total_voxels = x_points * y_points * z_points
        
        print(f"Adjusted resolution - lateral: {lateral_res:.2f}, axial: {axial_res:.2f}")
        print(f"New grid dimensions: {x_points} x {y_points} x {z_points} = {total_voxels} voxels")
    
    # Create coordinate vectors for the grid
    x = np.linspace(x_min, x_max, x_points)
    y = np.linspace(y_min, y_max, y_points)
    z = np.linspace(z_min, z_max, z_points)
    
    # Pre-allocate the voxel grid
    voxels = np.zeros((x_points, y_points, z_points), dtype=bool)
    
    # Process one Z-slice at a time to reduce memory usage
    print("Starting voxelization (processing by Z-slices)...")
    for z_idx, z_val in enumerate(tqdm(z)):
        # Get memory usage for monitoring
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1024**2  # in MB
        if mem_usage > 1000:  # If using more than 1GB
            print(f"Warning: High memory usage ({mem_usage:.1f} MB)")
        
        # Create a 2D grid of points for this Z-slice
        X, Y = np.meshgrid(x + lateral_res/2, y + lateral_res/2, indexing='ij')
        points = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, z_val + axial_res/2)))
        
        # Process in batches
        inside = np.empty(points.shape[0], dtype=bool)
        num_points = points.shape[0]
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            inside[start:end] = mesh.contains(points[start:end])
        
        # Update the voxel grid for this Z-slice
        voxels[:, :, z_idx] = inside.reshape(X.shape)
    
    return voxels, (x, y, z)

def visualize_voxels(voxels, grid=None, voxel_aspect=(8, 8, 7), max_display_voxels=100000):
    """
    Visualize a 3D voxel grid using matplotlib's voxel plotting.
    
    Parameters:
      voxels (numpy.ndarray): 3D boolean array representing the voxelized model.
      grid (tuple): Optional tuple of (x, y, z) arrays representing the voxel grid coordinates.
      voxel_aspect (tuple): Relative aspect ratio of the voxels (lateral, lateral, axial).
      max_display_voxels (int): Maximum number of voxels to display to prevent rendering issues.
    """
    # Count the number of active voxels
    active_voxels = np.sum(voxels)
    print(f"Total voxels: {voxels.size}, Active voxels: {active_voxels}")
    
    # Check if there are too many voxels to visualize
    if active_voxels > max_display_voxels:
        print(f"Warning: Too many active voxels ({active_voxels}) to visualize effectively.")
        print(f"Downsampling for visualization...")
        
        # Downsample the voxel grid for visualization
        downsample_factor = int(np.ceil((active_voxels / max_display_voxels) ** (1/3)))
        downsampled = voxels[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled to {np.sum(downsampled)} voxels for visualization.")
        voxels = downsampled
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the voxels with edge color to delineate voxel boundaries
    ax.voxels(voxels, edgecolor='k', alpha=0.5)
    
    # Set the aspect ratio based on the voxel sizes
    ax.set_box_aspect(voxel_aspect)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Simulated SD-OCT Voxel Image")
    plt.tight_layout()
    plt.show()

def save_voxels_as_slices(voxels, output_dir='oct_slices'):
    """
    Save the 3D voxel array as a series of 2D slice images.
    
    Parameters:
      voxels (numpy.ndarray): 3D boolean array representing the voxelized model.
      output_dir (str): Directory to save the slice images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each Z-slice as a separate image
    for z in range(voxels.shape[2]):
        plt.figure(figsize=(8, 8))
        plt.imshow(voxels[:, :, z].T, cmap='gray', interpolation='none')
        plt.title(f"Z-slice {z}")
        plt.colorbar(label="Density")
        plt.savefig(f"{output_dir}/slice_{z:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {voxels.shape[2]} slices to {output_dir}/")

if __name__ == '__main__':
    # Path to your OBJ file
    obj_file = '/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj'
    
    # You may need to adjust these parameters based on your model size
    try:
        voxels, grid = simulate_sd_oct(
            obj_file,
            axial_res=10,  # Increased resolution to reduce voxel count
            lateral_res=12, # Increased resolution to reduce voxel count
            batch_size=50000,
            max_voxels=10000000  # Limit total voxels to prevent memory issues
        )
        
        # Save the voxel grid as a compressed array to save disk space
        print("Saving voxel data...")
        np.savez_compressed('voxel_image.npz', voxels=voxels)
        
        # Visualize the voxelized model
        visualize_voxels(voxels, grid, voxel_aspect=(12, 12, 10))
        
        # Optionally save as 2D slices
        save_voxels_as_slices(voxels)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Try reducing the resolution or using a smaller model.")