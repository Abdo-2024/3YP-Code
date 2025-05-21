import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import csv
import time
start_time = time.time()
"""
This script loads and preprocesses a CAD STL and a scanned mesh, classifies scan points as inside or outside the CAD volume, removes points near the CAD surface, and clusters the remaining points using DBSCAN. It fits spheres to each cluster, labels them based on inside/outside majority, and visualises the scan mesh with coloured spheres in PyVista. Finally, it exports cluster statistics to a CSV and reports execution time.
"""

############################################################################
# 1. Load & Preprocess
############################################################################
cad_mesh_path ="/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl"
scan_mesh_path = "/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj"

cad_mesh = pv.read(cad_mesh_path)
scan_mesh = pv.read(scan_mesh_path)

# Process the CAD mesh
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)
cad_mesh = cad_mesh.triangulate().decimate(0.8)

# Process the scanned mesh
scan_mesh = scan_mesh.clean(tolerance=1e-15).fill_holes(hole_size=2).triangulate().decimate(0.8)

############################################################################
# 2. Fast inside/outside classification with select_enclosed_points
############################################################################
select = scan_mesh.select_enclosed_points(cad_mesh)
inside_out_array = select["SelectedPoints"]  # 1 for inside, 0 for outside

############################################################################
# 3. Remove points near/on the CAD surface
############################################################################
# Build a KDTree from the CAD mesh points (not the scan points)
scan_points = scan_mesh.points
cad_tree = cKDTree(cad_mesh.points)
dists, _ = cad_tree.query(scan_points)
surface_tolerance = 0.2
mask = dists >= surface_tolerance  # keep points not near the CAD surface
print(f"Removing {np.count_nonzero(~mask)} points near the CAD surface...")

filtered_scan_points = scan_points[mask]
filtered_inout = inside_out_array[mask]

############################################################################
# 4. Cluster (“blob”) detection with DBSCAN on filtered scan points
############################################################################
# Adjust eps and min_samples as needed for your data
db = DBSCAN(eps=9, min_samples=10)
labels = db.fit_predict(filtered_scan_points)
unique_clusters = np.unique(labels)

############################################################################
# 5. Majority inside vs. outside per cluster and sphere fitting
############################################################################
def fit_sphere(points):
    """
    Returns (center, radius) for a sphere fitted to 'points'
    using the mean as the center and max distance as the radius.
    """
    if len(points) == 0:
        return np.array([0, 0, 0]), 0.0
    center = np.mean(points, axis=0)
    diffs = points - center
    radius = np.sqrt((diffs**2).sum(axis=1).max())
    return center, radius

cluster_info = []
for cid in unique_clusters:
    if cid == -1:
        # DBSCAN labels noise as -1
        continue
    cluster_mask = (labels == cid)
    cluster_pts = filtered_scan_points[cluster_mask]
    cluster_inout = filtered_inout[cluster_mask]
    
    if len(cluster_pts) == 0:
        continue

    # Count inside vs. outside points in the cluster
    inside_count = np.count_nonzero(cluster_inout == 1)
    outside_count = np.count_nonzero(cluster_inout == 0)

    if inside_count > outside_count:
        label = "subtraction"
    elif inside_count == outside_count:
        label = "green"
    else:
        label = "addition"

    center, radius = fit_sphere(cluster_pts)
    
    cluster_info.append({
        "cluster_id": cid,
        "points_in_cluster": len(cluster_pts),
        "inside_count": inside_count,
        "outside_count": outside_count,
        "label": label,
        "center": center,
        "radius": radius
    })

############################################################################
# 6. Visualization in PyVista
############################################################################
p = pv.Plotter()

# Display the CAD mesh (semi-transparent white)
# p.add_mesh(cad_mesh, color="white", opacity=0.5, show_edges=False, label="CAD Mesh")

# Display the full scanned mesh in light gray (optional)
p.add_mesh(scan_mesh, color="lightgray", opacity=1, show_edges=False, label="Scan Mesh")

# Draw a sphere for each detected cluster colored by its label:
for info in cluster_info:
    c = info["center"]
    r = info["radius"]
    lbl = info["label"]
    
    sphere = pv.Sphere(radius=r, center=c, theta_resolution=15, phi_resolution=15)
    
    if lbl == "addition":
        color = "red"
    elif lbl == "subtraction":
        color = "blue"
    else:
        color = "green"
    
    p.add_mesh(sphere, color=color, opacity=0.5)
    # Optionally, add a text label at the sphere center:
    # p.add_point_labels([c], [f"{lbl} cluster {info['cluster_id']}"], font_size=12)

# Add a legend
p.add_legend([
    ["CAD Mesh", "white"],
    ["Scan Mesh", "lightgray"],
    ["Addition Blob", "red"],
    ["Subtraction Blob", "blue"],
    ["Tie Blob", "green"]
], bcolor="white")

p.show()

############################################################################
# 7. (Optional) Output cluster summary
############################################################################
#for info in cluster_info:
#    cid = info["cluster_id"]
#    lab = info["label"]
#    ni  = info["inside_count"]
#    no  = info["outside_count"]
#    print(f"Cluster {cid}: label={lab}, inside={ni}, outside={no}, center={info['center']}, radius={info['radius']:.2f}")

############################################################################
# 8. Export to CSV
############################################################################
with open("cluster_summary.csv", "w", newline="") as csvfile:
    fieldnames = [
        "cluster_id",
        "x", "y", "z",
        "radius",
        "label",
        "inside_count",
        "outside_count",
        "points_in_cluster"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for info in cluster_info:
        writer.writerow({
            "cluster_id": info["cluster_id"],
            "x": info["center"][0],
            "y": info["center"][1],
            "z": info["center"][2],
            "radius": info["radius"],
            "label": info["label"],
            "inside_count": info["inside_count"],
            "outside_count": info["outside_count"],
            "points_in_cluster": info["points_in_cluster"]
        })
end_time = time.time()
print("Execution time: {:.4f} seconds".format(end_time - start_time))
print("Cluster summary has been exported to cluster_summary.csv")

