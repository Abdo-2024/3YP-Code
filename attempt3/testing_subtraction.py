import trimesh
import numpy as np
import random

# Load and check the STL model
stl_model = trimesh.load_mesh("/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model.stl")

# Repair the model if it's not watertight
if not stl_model.is_watertight:
    stl_model.fill_holes()

# Function to generate a random sphere
def generate_random_sphere(radius_range=(1, 5), position_range=(-1, 100)):
    radius = random.uniform(*radius_range)
    position = np.array([random.uniform(*position_range) for _ in range(3)])
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    sphere.apply_translation(position)
    
    # Repair the sphere if it's not watertight
    if not sphere.is_watertight:
        sphere.fill_holes()
    
    return sphere

# Generate 10 random spheres
spheres = [generate_random_sphere() for _ in range(10)]

# Subtract the spheres from the STL model
for sphere in spheres:
    stl_model = stl_model.difference(sphere)

# Save the resulting model
stl_model.export("swiss_cheese_model2.stl")
