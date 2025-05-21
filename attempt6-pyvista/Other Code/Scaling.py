from stl import mesh  # Import the mesh module from the stl package

def scale_stl(file_path, output_path, scale_factor):
    """
    Scale the STL file by a given factor.
    
    Args:
        file_path (str): Path to the original STL file.
        output_path (str): Path to save the scaled STL file.
        scale_factor (float): Factor to scale the STL model.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)  # Use the provided file path
    
    # Scale the vertices
    stl_mesh.vectors *= scale_factor  # Multiply each vertex by the scale factor
    
    # Save the scaled STL file
    stl_mesh.save(output_path)
    print(f"Scaled STL file saved at: {output_path}")

if __name__ == "__main__":
    # Usage example
    original_file_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Casing.stl"  # Path to the original (incorrectly scaled) STL file
    scaled_file_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/4x4_NEW_MN_um_scale.stl"  # Path to save the corrected (scaled up) STL file
    
    # Scale factor to reverse the earlier downscaling by 1000
    scale_factor = 1000 

    # Call the function to scale the STL
    scale_stl(original_file_path, scaled_file_path, scale_factor)
