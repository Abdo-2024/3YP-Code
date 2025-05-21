import pymesh

def subtract_meshes_pymesh(mesh1_path, mesh2_path, output_path, engine="cgal"):
    """
    Subtract mesh2 from mesh1 using PyMesh's boolean operations and export the result as an STL file.
    
    Parameters:
      mesh1_path (str): Path to the main STL file (the mesh to subtract from).
      mesh2_path (str): Path to the STL file to subtract.
      output_path (str): Path where the resulting mesh will be saved.
      engine (str): Boolean engine to use ("cgal", "igl", or "cork").
    """
    # Load the meshes from the given STL files.
    mesh1 = pymesh.load_mesh(mesh1_path, '/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl')
    mesh2 = pymesh.load_mesh(mesh2_path, '/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled+Noise.stl')
    
    # Clean up meshes by removing duplicated vertices (tolerance 1e-6).
    mesh1, _ = pymesh.remove_duplicated_vertices(mesh1, 1e-6)
    mesh2, _ = pymesh.remove_duplicated_vertices(mesh2, 1e-6)
    
    # Perform boolean subtraction (difference operation).
    result_mesh = pymesh.boolean(mesh1, mesh2, operation="difference", engine=engine)
    
    # Save the resulting mesh as an STL file.
    pymesh.save_mesh(output_path, result_mesh)
    print(f"Resulting mesh saved as {output_path}")

# Example usage:
subtract_meshes_pymesh("mesh1.stl", "mesh2.stl", "subtracted_mesh.stl", engine="cgal")
