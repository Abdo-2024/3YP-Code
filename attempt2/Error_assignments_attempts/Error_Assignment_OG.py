import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
import csv

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
# Load the original and deformed STL meshes.
mesh_original = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl")
mesh_deformed = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed2.stl")

# Ensure the meshes have vertex normals (for better ICP registration and visualisation)
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# Sample points from the surface of each mesh.
n_points = 50000
pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

# ---------------------------
# Step 2: Align the Two Models using ICP
# ---------------------------
threshold_icp = 0.02  # distance threshold for ICP
trans_init = np.eye(4)  # initial transformation (identity)

reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_deformed, pcd_original, threshold_icp, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP Transformation:\n", reg_p2p.transformation)

# Apply the transformation to the deformed point cloud.
pcd_deformed.transform(reg_p2p.transformation)

# ---------------------------
# Step 3: Subtract the Original from the Deformed to Extract Noise Points
# ---------------------------
# For each point in the deformed point cloud, compute its distance to the nearest point in the original cloud.
original_kd_tree = o3d.geometry.KDTreeFlann(pcd_original)
deformation_points = []

# Define a threshold distance above which the point is considered to be deformed.
deformation_threshold = 0.01  # Adjust as necessary

for point in np.asarray(pcd_deformed.points):
    [k, idx, _] = original_kd_tree.search_knn_vector_3d(point, 1)
    nearest_point = np.asarray(pcd_original.points)[idx[0]]
    distance = np.linalg.norm(point - nearest_point)
    if distance > deformation_threshold:
        deformation_points.append(point)

deformation_points = np.array(deformation_points)
print("Number of deformation points:", len(deformation_points))

if len(deformation_points) == 0:
    print("No deformation points detected. Please adjust the deformation threshold or check your meshes.")
    # Optionally, exit the programme:
    # import sys
    # sys.exit()

# ---------------------------
# Step 4: Cluster the Deformation Points into Blobs using DBSCAN
# ---------------------------
# Adjusted DBSCAN parameters for larger blobs:
dbscan = DBSCAN(eps=0.001, min_samples=5)
labels = dbscan.fit_predict(deformation_points)

# Identify unique clusters (exclude noise, which is labelled as -1).
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)

print("Detected blobs (clusters):", len(unique_labels))

# If no clusters are detected, consider visualising the deformation points:
if len(unique_labels) == 0:
    print("No clusters detected. You may want to visualise the deformation points to adjust DBSCAN parameters.")
    deformation_pcd = o3d.geometry.PointCloud()
    deformation_pcd.points = o3d.utility.Vector3dVector(deformation_points)
    o3d.visualization.draw_geometries([deformation_pcd])
    # Optionally, exit if clustering is critical:
    # import sys
    # sys.exit()

# ---------------------------
# Step 5: Fit a Sphere to Each Blob and Compute Deformation Intensity
# ---------------------------
def sphere_residuals(params, points):
    """
    Compute residuals between the distances of points from the sphere centre and the sphere radius.
    params: (a, b, c, r) where (a, b, c) is the centre and r is the radius.
    """
    a, b, c, r = params
    residuals = np.sqrt((points[:, 0] - a)**2 + (points[:, 1] - b)**2 + (points[:, 2] - c)**2) - r
    return residuals

def fit_sphere_to_points(points):
    # Initial guess: centre at the centroid and radius as the mean distance.
    centroid = np.mean(points, axis=0)
    r_initial = np.mean(np.linalg.norm(points - centroid, axis=1))
    initial_guess = np.append(centroid, r_initial)
    
    # Use non-linear least squares optimisation.
    result = least_squares(sphere_residuals, initial_guess, args=(points,))
    a, b, c, r = result.x
    return (np.array([a, b, c]), abs(r))  # Ensure radius is positive

blobs = {}
# Define thresholds for sphere radius colouring (adjust as appropriate).
threshold_low = 0.005
threshold_high = 0.01

# Prepare a list for CSV output.
csv_data = []

for label in unique_labels:
    # Extract points for the current blob.
    blob_points = deformation_points[labels == label]
    
    # Fit a sphere to the blob points.
    centre, radius = fit_sphere_to_points(blob_points)
    
    # Compute deformation intensity as the mean absolute residual of distances from the sphere.
    distances = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances - radius)
    deformation_intensity = np.mean(residuals)
    
    # Colour the sphere based on its radius.
    if radius > threshold_high:
        colour = [1, 0, 0]   # Red
    elif radius < threshold_low:
        colour = [1, 1, 0]   # Yellow
    else:
        colour = [1, 0.65, 0]  # Orange
    
    blobs[label] = {
        "centre": centre,
        "radius": radius,
        "colour": colour,
        "deformation_intensity": deformation_intensity
    }
    
    print(f"Blob {label}: Centre = {centre}, Radius = {radius:.6f}, Deformation Intensity = {deformation_intensity:.6f}, Colour = {colour}")
    
    # Append blob data for CSV output.
    csv_data.append({
        'x': centre[0],
        'y': centre[1],
        'z': centre[2],
        'radius': radius,
        'deformation_intensity': deformation_intensity
    })

# Write the sphere data to a CSV file.
csv_filename = "sphere_data_detection.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)
print(f"Sphere data written to {csv_filename}")

# ---------------------------
# Step 6: Visualise the Original Model and Overlaid Spheres
# ---------------------------
spheres = []

for label, sphere_params in blobs.items():
    centre = sphere_params["centre"]
    radius = sphere_params["radius"]
    colour = sphere_params["colour"]
    
    # Create a sphere mesh for visualisation.
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere_mesh.translate(centre)
    
    # Set sphere colour.
    sphere_mesh.paint_uniform_color(colour)
    
    spheres.append(sphere_mesh)

# Prepare visualisation objects.
geometries = [mesh_original, mesh_deformed] + spheres

# Launch the visualiser.
o3d.visualization.draw_geometries(geometries)
