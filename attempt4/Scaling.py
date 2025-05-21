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
    stl_mesh.vectors *= scale_factor
    
    # Save the scaled STL file
    stl_mesh.save(output_path)
    print(f"Scaled STL file saved at: {output_path}")

if __name__ == "__main__":
    # Usage example
    original_file_path = "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/1250_polygon_sphere_100mm.STL"  # Replace with actual path
    scaled_file_path = "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model.stl"  # Replace with desired path
    scale_factor = 1 / 1000  # Scale down by 1000

    scale_stl(original_file_path, scaled_file_path, scale_factor)
