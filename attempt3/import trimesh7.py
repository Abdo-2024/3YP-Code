import trimesh
import numpy as np

def add_random_holes(mesh, num_holes=5, hole_radius_range=(0.01, 0.05)):
    """
    Introduces random spherical holes into the mesh to simulate imperfections.

    Parameters:
    - mesh: Trimesh object representing the 3D model.
    - num_holes: Number of holes to add.
    - hole_radius_range: Tuple specifying the minimum and maximum radius of the holes.
    """
    # Check if the mesh is watertight
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fix it...")
        mesh = mesh.fill_holes()

    if not mesh.is_watertight:
        print("Mesh still not watertight after filling holes!")
        return mesh

    for _ in range(num_holes):
        # Generate random position for the hole center within the mesh bounds
        min_bound, max_bound = mesh.bounds
        hole_center = np.random.uniform(min_bound, max_bound)
        
        # Generate a random radius for the hole
        hole_radius = np.random.uniform(*hole_radius_range)
        
        # Create a sphere mesh representing the hole
        hole = trimesh.creation.icosphere(radius=hole_radius, subdivisions=3)
        hole.apply_translation(hole_center)
        
        # Perform a boolean subtraction to remove the hole from the mesh
        mesh = mesh.difference(hole)
    
    return mesh

def main():
    # Load your STL file
    mesh = trimesh.load_mesh('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model.stl')
    
    # Check if the mesh is loaded correctly
    if mesh.is_empty:
        print("Failed to load the mesh.")
        return
    
    # Add random holes to the mesh
    modified_mesh = add_random_holes(mesh)
    
    # Export the modified mesh to a new STL file
    modified_mesh.export('modified_model_with_holes.stl')

if __name__ == "__main__":
    main()
