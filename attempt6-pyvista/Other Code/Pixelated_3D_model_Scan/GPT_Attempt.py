from __future__ import annotations
import pyvista as pv  # type: ignore
import numpy as np
import random
import csv

# ------------------------------------------------------------------------------
# Step 1: Load and Scale Up the Main STL Model (Fix for Numerical Issues)
# ------------------------------------------------------------------------------
SCALE_FACTOR = 1000  # Scale up from µm to mm

main_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN.stl")  # Update path if needed
main_mesh.scale(SCALE_FACTOR, inplace=True)  # Scale UP
main_mesh = main_mesh.triangulate().clean()
main_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

cell_centers = main_mesh.cell_centers()
xmin, xmax, ymin, ymax, zmin, zmax = main_mesh.bounds
margin = (zmax - zmin) * 0.05  # Avoid the very bottom region

# ------------------------------------------------------------------------------
# Step 2: Generate Addition & Subtraction Spheres
# ------------------------------------------------------------------------------
num_addition_spheres = 5
num_subtraction_spheres = 5

addition_spheres = []
subtraction_spheres = []
addition_data = []
subtraction_data = []

def is_fully_inside(mesh, candidate):
    sel = mesh.select_enclosed_points(candidate, tolerance=0.0)
    return np.all(sel.point_data["SelectedPoints"])

# Generate addition spheres
for _ in range(num_addition_spheres):
    while True:
        idx = random.randint(0, cell_centers.n_points - 1)
        point = cell_centers.points[idx]
        if point[2] <= zmin + margin:
            continue
        
        normal = main_mesh.cell_normals[idx]
        radius = random.uniform(0.015, 0.02) * SCALE_FACTOR  # Scale sphere size
        sphere = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20)
        sphere.triangulate().clean()

        offset = random.uniform(0.2 * radius, 0.8 * radius)
        sphere_center = point + normal * offset
        sphere.translate(sphere_center, inplace=True)

        if is_fully_inside(main_mesh, sphere):
            continue

        addition_spheres.append(sphere)
        addition_data.append({"center": sphere_center, "radius": radius})
        break

# Generate subtraction spheres
overlap_factor = 1.1
for _ in range(num_subtraction_spheres):
    while True:
        idx = random.randint(0, cell_centers.n_points - 1)
        point = cell_centers.points[idx]
        if point[2] <= zmin + margin:
            continue

        normal = main_mesh.cell_normals[idx]
        radius = random.uniform(0.015, 0.02) * SCALE_FACTOR  # Scale sphere size
        sphere = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20)
        sphere.triangulate().clean()

        offset = random.uniform(0.2 * radius, 0.8 * radius)
        sphere_center = point + normal * offset

        # Ensure subtraction spheres do not overlap with addition spheres
        conflict = any(
            np.linalg.norm(sphere_center - data["center"]) < (radius + data["radius"]) * overlap_factor
            for data in addition_data
        )
        if conflict:
            continue

        sphere.translate(sphere_center, inplace=True)

        if is_fully_inside(main_mesh, sphere):
            continue

        subtraction_spheres.append(sphere)
        subtraction_data.append({"center": sphere_center, "radius": radius})
        break

# ------------------------------------------------------------------------------
# Step 3: Boolean Operations (Addition & Subtraction)
# ------------------------------------------------------------------------------
model_with_addition = main_mesh.copy()

for sphere in addition_spheres:
    intersection_mesh = model_with_addition.intersection(sphere, split_first_output=False)

    if intersection_mesh is None or intersection_mesh.n_points == 0:
        continue

    try:
        model_with_addition = model_with_addition.boolean_union(sphere, tolerance=1e-6)
    except Exception as e:
        print("Error during union of addition sphere:", e)
        continue

final_mesh = model_with_addition.copy()

for sphere in subtraction_spheres:
    intersection_mesh = final_mesh.intersection(sphere, split_first_output=False)

    if intersection_mesh is None or intersection_mesh.n_points == 0:
        continue

    try:
        final_mesh = final_mesh.boolean_difference(sphere, tolerance=1e-6)
    except Exception as e:
        print("Error during boolean difference:", e)
        continue

# ------------------------------------------------------------------------------
# Step 4: Scale Down and Save (Fix for Returning to Original Size)
# ------------------------------------------------------------------------------
final_mesh.scale(1 / SCALE_FACTOR, inplace=True)  # Scale BACK to original µm size
final_mesh.save("2x2_MN_w_Noise.stl")
final_mesh.plot(color="lightblue")

# ------------------------------------------------------------------------------
# Step 5: Export Sphere Data to CSV
# ------------------------------------------------------------------------------
csv_filename = "sphere_data.csv"
with open(csv_filename, mode="w", newline="") as csv_file:
    fieldnames = ["set", "x", "y", "z", "radius"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for data in addition_data:
        writer.writerow({
            "set": "addition",
            "x": data["center"][0] / SCALE_FACTOR,  # Convert back to µm
            "y": data["center"][1] / SCALE_FACTOR,
            "z": data["center"][2] / SCALE_FACTOR,
            "radius": data["radius"] / SCALE_FACTOR
        })

    for data in subtraction_data:
        writer.writerow({
            "set": "subtraction",
            "x": data["center"][0] / SCALE_FACTOR,
            "y": data["center"][1] / SCALE_FACTOR,
            "z": data["center"][2] / SCALE_FACTOR,
            "radius": data["radius"] / SCALE_FACTOR
        })
