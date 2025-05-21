from __future__ import annotations

import pyvista as pv  # type: ignore
import numpy as np
import random
import csv

def deform_sphere(sphere, num_blobs=1, blob_radius=0.05, blob_intensity=0.01):
    """
    Deform a PyVista sphere to create lumps by randomly displacing its points.
    """
    sphere = sphere.triangulate().clean()
    points = sphere.points.copy()
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    blob_centres = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    for i, pt in enumerate(points):
        total_disp = np.zeros(3)
        for centre in blob_centres:
            direction = pt - centre
            distance = np.linalg.norm(direction)
            if distance < blob_radius:
                factor = 1 - (distance / blob_radius)
                if distance != 0:
                    unit_direction = direction / distance
                else:
                    unit_direction = np.zeros(3)
                total_disp += blob_intensity * factor * unit_direction
        points[i] = pt + total_disp

    sphere.points = points
    return sphere

def is_fully_inside(mesh, candidate, tol=0.0):
    """
    Return True if 'candidate' is fully inside 'mesh', using 'select_enclosed_points'.
    We keep the default tolerance small (e.g., 0.0).
    """
    sel = mesh.select_enclosed_points(candidate, tolerance=tol)
    return np.all(sel.point_data["SelectedPoints"])


# ------------------------------------------------------------------------------
# Step 1: Load and scale the main STL model.
# ------------------------------------------------------------------------------
main_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/1_Needle.stl")
# Triangulate and clean
main_mesh = main_mesh.triangulate().clean()
main_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

# If your model is extremely small (µm scale), scale it up:
scale_factor = 1000.0
# main_mesh.points *= scale_factor

# Compute cell centres for sampling
cell_centers = main_mesh.cell_centers()

xmin, xmax, ymin, ymax, zmin, zmax = main_mesh.bounds
margin = (zmax - zmin) * 0.05  # Avoid the very bottom region

# ------------------------------------------------------------------------------
# Step 2: Generate spheres.
# ------------------------------------------------------------------------------
num_addition_spheres = 5
num_subtraction_spheres = 5
overlap_factor = 1.1

addition_spheres = []
addition_data = []
subtraction_spheres = []
subtraction_data = []

for i in range(num_addition_spheres):
    while True:
        idx = random.randint(0, cell_centers.n_points - 1)
        point = cell_centers.points[idx]

        # Keep clear of the very bottom
        if point[2] <= zmin + margin:
            continue

        normal = main_mesh.cell_normals[idx]
        radius = random.uniform(20, 50)  # This is also in scaled units
        sphere = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20).triangulate().clean()
        
        # Offset from surface
        offset = random.uniform(0.5 * radius, 0.8 * radius)
        sphere_center = point + normal * offset
        sphere.translate(sphere_center, inplace=True)
        
        # Check if fully inside the main mesh
        if is_fully_inside(main_mesh, sphere, tol=0.0):
            continue
        
        # Deform
        random_intensity_factor = random.uniform(0.01, 0.05)
        deformation_intensity = random_intensity_factor * radius
        deformed_sphere = deform_sphere(
            sphere,
            num_blobs=20,
            blob_radius=radius * 1.3,
            blob_intensity=deformation_intensity
        )

        addition_spheres.append(deformed_sphere)
        addition_data.append({"center": sphere_center.copy(), "radius": radius})
        break

for i in range(num_subtraction_spheres):
    while True:
        idx = random.randint(0, cell_centers.n_points - 1)
        point = cell_centers.points[idx]
        if point[2] <= zmin + margin:
            continue
        
        normal = main_mesh.cell_normals[idx]
        radius = random.uniform(20, 50)
        sphere = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20).triangulate().clean()
        
        offset = random.uniform(0.5 * radius, 0.8 * radius)
        sphere_center = point + normal * offset
        
        # Check that the new sphere doesn't intersect an addition sphere too closely
        conflict = False
        for data in addition_data:
            dist = np.linalg.norm(sphere_center - data["center"])
            if dist < (radius + data["radius"]) * overlap_factor:
                conflict = True
                break
        if conflict:
            continue

        sphere.translate(sphere_center, inplace=True)
        
        if is_fully_inside(main_mesh, sphere, tol=0.0):
            continue
        
        # Deform
        random_intensity_factor = random.uniform(0.01, 0.05)
        deformation_intensity = random_intensity_factor * radius
        deformed_sphere = deform_sphere(
            sphere,
            num_blobs=20,
            blob_radius=radius * 1.2,
            blob_intensity=deformation_intensity
        )
        subtraction_spheres.append(deformed_sphere)
        subtraction_data.append({"center": sphere_center, "radius": radius})
        break

# ------------------------------------------------------------------------------
# Optional Debugging Visualisation
# ------------------------------------------------------------------------------
# If you'd like, you can uncomment these lines to see the scaled model visually:
pl = pv.Plotter()
pl.add_mesh(main_mesh, color="white", opacity=0.8, label="Main Model (scaled)")
for s in addition_spheres:
   pl.add_mesh(s, color="blue")
for s in subtraction_spheres:
   pl.add_mesh(s, color="red")
pl.add_legend()
pl.show()

# ------------------------------------------------------------------------------
# Step 3: Boolean operations with safer checks
# ------------------------------------------------------------------------------
model_with_addition = main_mesh.copy()


def safe_union(base_mesh, sphere):
    """
    Attempt a boolean union of 'sphere' with 'base_mesh'.
    If intersection is present but fails, skip. If no intersection, try anyway.
    """
    result = base_mesh.intersection(sphere)
    # If intersection(...) returned a tuple, take the first part
    if isinstance(result, tuple):
        imesh = result[0]
    else:
        imesh = result

    if imesh is None or imesh.n_points == 0:
        # No real intersection, but we can still union if it doesn't fail
        try:
            return base_mesh.boolean_union(sphere, tolerance=1e-3)
        except Exception as e:
            print("Union failed on a non-intersecting sphere:", e)
            return base_mesh
    else:
        # We do have an intersection
        try:
            return base_mesh.boolean_union(sphere, tolerance=1e-6)
        except Exception as e:
            print("Union failed on an intersecting sphere:", e)
            return base_mesh


def safe_difference(base_mesh, sphere):
    """
    Attempt a boolean difference to subtract 'sphere' from 'base_mesh'.
    If there's no intersection or if the operation fails, skip it.
    """
    result = base_mesh.intersection(sphere)
    if isinstance(result, tuple):
        imesh = result[0]
    else:
        imesh = result

    if imesh is None or imesh.n_points == 0:
        # No intersection => no subtraction
        return base_mesh
    else:
        try:
            return base_mesh.boolean_difference(sphere, tolerance=1e-3)
        except Exception as e:
            print("Difference failed on a sphere:", e)
            return base_mesh

# Perform additions
for sphere in addition_spheres:
    # Triangulate and clean each time
    sphere = sphere.triangulate().clean()
    model_with_addition = safe_union(model_with_addition, sphere)

final_mesh = model_with_addition.copy()

# Perform subtractions
for sphere in subtraction_spheres:
    sphere = sphere.triangulate().clean()
    final_mesh = safe_difference(final_mesh, sphere)

# ------------------------------------------------------------------------------
# Step 3.1: Export sphere data
# ------------------------------------------------------------------------------
csv_filename = "sphere_data.csv"
with open(csv_filename, mode="w", newline="") as csv_file:
    fieldnames = ["set", "x", "y", "z", "radius"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for data in addition_data:
        writer.writerow({
            "set": "addition",
            "x": data["center"][0],
            "y": data["center"][1],
            "z": data["center"][2],
            "radius": data["radius"]
        })
    for data in subtraction_data:
        writer.writerow({
            "set": "subtraction",
            "x": data["center"][0],
            "y": data["center"][1],
            "z": data["center"][2],
            "radius": data["radius"]
        })

# ------------------------------------------------------------------------------
# Step 4: Scale down the final mesh if you want to return to the original size
# ------------------------------------------------------------------------------
# final_mesh.points /= scale_factor

# ------------------------------------------------------------------------------
# Finally, save and display
# ------------------------------------------------------------------------------
final_mesh.save("2x2_MN_w_Noise.stl")
final_mesh.plot(color="lightblue")
