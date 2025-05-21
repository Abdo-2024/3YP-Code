import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

# 1. Load STL
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN+Noise.stl")

# 2. Clean
scan_mesh = scan_mesh.clean(tolerance=1e-5)
scan_mesh = scan_mesh.fill_holes(hole_size=10)
scan_mesh = scan_mesh.triangulate()

# 3. Decimate (reduce triangles by e.g. 80%)
cad_mesh = cad_mesh.decimate(0.8)
scan_mesh = scan_mesh.decimate(0.8)

# 4. Sub-sample vertices if still too large
def random_subsample_points(mesh, max_points):
    pts = mesh.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx]
    else:
        return pts

max_points = 20000
cad_points = random_subsample_points(cad_mesh, max_points)
scan_points = random_subsample_points(scan_mesh, max_points)

# 5. KD-tree & Distances
cad_tree = cKDTree(cad_points)
scan_tree = cKDTree(scan_points)

dist_scan_to_cad, _ = cad_tree.query(scan_points)
dist_cad_to_scan, _ = scan_tree.query(cad_points)

# 6. Outliers
threshold = 11
extra_idx = np.where(dist_scan_to_cad > threshold)[0]
missing_idx = np.where(dist_cad_to_scan > threshold)[0]

extra_points = scan_points[extra_idx]
missing_points = cad_points[missing_idx]

# 7. (Optional) Clustering
def cluster_outliers(points, eps=0.5, min_samples=10):
    if len(points) == 0:
        return np.array([]), []
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels, set(labels)

labels_extra, unique_extra = cluster_outliers(extra_points)
labels_missing, unique_missing = cluster_outliers(missing_points)

# 8. Visualization with point glyphs
plotter = pv.Plotter()
# plotter.add_mesh(cad_mesh, color="lightyellow", opacity=0.1)
plotter.add_mesh(scan_mesh, color="lightgreen", opacity=0.5)

# Show outliers as simple points (much faster than spheres)
plotter.add_points(extra_points, color="red", point_size=5, render_points_as_spheres=True)
plotter.add_points(missing_points, color="blue", point_size=5, render_points_as_spheres=True)

plotter.show()
