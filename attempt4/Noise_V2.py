import pymesh
import numpy as np
import random
"""
This script loads a 3D mesh using PyMesh and procedurally generates two sets of deformed icospheres: one to add noise and another to subtract material. It samples random points within the mesh’s bounding box (excluding the bottom margin), deforms each sphere by applying random vertex displacements, and then uses Boolean CSG operations—union for noise addition, intersection with the main mesh, and difference for material removal—to produce a modified mesh. The final result is saved as an STL file.
"""

# ---------------------------------------------------------------------------
# Helper: a simple deformation function.
def deform_mesh(mesh, intensity):
    # For each vertex, add a small random displacement scaled by intensity.
    vertices = mesh.vertices.copy()
    for i, v in enumerate(vertices):
        # Generate a displacement in range [-0.5, 0.5] scaled by intensity.
        displacement = intensity * (np.random.rand(3) - 0.5)
        vertices[i] = v + displacement
    return pymesh.form_mesh(vertices, mesh.faces)

# ---------------------------------------------------------------------------
# Parameters
#main_stl_path = "path/to/your/model.stl"  # Use your STL file path

# Parameters for noise spheres (addition)
num_add_spheres = 25
add_radius_range = (0.005, 0.015)
add_intensity_factor_range = (0.01, 0.5)
add_refine_order = 2  # Refinement order for generated icospheres

# Parameters for subtraction spheres
num_sub_spheres = 15
sub_radius_range = (0.005, 0.015)
sub_intensity_factor_range = (0.01, 0.5)
sub_refine_order = 2

# ---------------------------------------------------------------------------
# Load main mesh directly with PyMesh use main_stl_path if you want to be able to change parameters in the above section 
main_mesh = pymesh.load_mesh("2x2_MN_Array_scaled.stl")

# Compute bounding box from the main mesh vertices.
min_coords = main_mesh.vertices.min(axis=0)
max_coords = main_mesh.vertices.max(axis=0)
margin = (max_coords[2] - min_coords[2]) * 0.05

# Lists to store the generated noise spheres.
addition_meshes = []
subtraction_meshes = []

# For simplicity, we sample random points inside the bounding box.
# (In a more refined version, you might sample points on the actual surface.)
def sample_point():
    p = np.random.uniform(min_coords, max_coords)
    while p[2] <= min_coords[2] + margin:
        p = np.random.uniform(min_coords, max_coords)
    return p

# ---------------------------------------------------------------------------
# Generate addition (noise) spheres
for i in range(num_add_spheres):
    center = sample_point()
    radius = random.uniform(*add_radius_range)
    # Choose an intensity proportional to the radius.
    intensity = random.uniform(*add_intensity_factor_range) * radius
    # Generate an icosphere using PyMesh.
    sphere = pymesh.generate_icosphere(radius, center=center, refinement_order=add_refine_order)
    # Deform the sphere to add noise.
    deformed = deform_mesh(sphere, intensity)
    addition_meshes.append(deformed)

# ---------------------------------------------------------------------------
# Generate subtraction spheres
# To reduce clustering, we use a basic rejection method.
sub_centers = []
min_distance = 2 * np.mean([sub_radius_range[0], sub_radius_range[1]])

for i in range(num_sub_spheres):
    attempt = 0
    valid = False
    while not valid and attempt < 100:
        p = sample_point()
        if all(np.linalg.norm(p - c) > min_distance for c in sub_centers):
            valid = True
            sub_centers.append(p)
        attempt += 1
    if not valid:
        sub_centers.append(p)
    radius = random.uniform(*sub_radius_range)
    intensity = random.uniform(*sub_intensity_factor_range) * radius
    sphere = pymesh.generate_icosphere(radius, center=p, refinement_order=sub_refine_order)
    deformed = deform_mesh(sphere, intensity)
    subtraction_meshes.append(deformed)

# ---------------------------------------------------------------------------
# Use PyMesh CSGTree to combine the meshes.
# First, union all addition spheres.
if len(addition_meshes) == 1:
    add_union = addition_meshes[0]
else:
    add_leaf_nodes = [{"mesh": m} for m in addition_meshes]
    add_tree = pymesh.CSGTree({"union": add_leaf_nodes})
    add_union = add_tree.mesh

# Intersect the union with the main mesh to "add" the noise.
add_result = pymesh.CSGTree({
    "intersection": [{"mesh": main_mesh}, {"mesh": add_union}]
}).mesh

# Next, union all subtraction spheres.
if len(subtraction_meshes) == 1:
    sub_union = subtraction_meshes[0]
else:
    sub_leaf_nodes = [{"mesh": m} for m in subtraction_meshes]
    sub_tree = pymesh.CSGTree({"union": sub_leaf_nodes})
    sub_union = sub_tree.mesh

# Finally, subtract the subtraction union from the addition result.
final_csg = pymesh.CSGTree({
    "difference": [{"mesh": add_result}, {"mesh": sub_union}]
})
final_mesh = final_csg.mesh

# ---------------------------------------------------------------------------
# Save the final modified mesh.
pymesh.save_mesh("modified_model.stl", final_mesh)
