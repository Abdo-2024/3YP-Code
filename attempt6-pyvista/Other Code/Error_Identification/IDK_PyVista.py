from __future__ import annotations

import pyvista as pv

# Load both meshes
cad_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/4x4_NEW_MN.stl")
scan_mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.stl")

# Process the CAD mesh (design)
cad_mesh = cad_mesh.compute_normals(auto_orient_normals=True)
cad_mesh = cad_mesh.triangulate()
#cad_mesh = cad_mesh.decimate(0.3)

# Process the scanned mesh
scan_mesh = scan_mesh.clean(tolerance=1e-15)
scan_mesh = scan_mesh.fill_holes(hole_size=2)
scan_mesh = scan_mesh.triangulate()
#scan_mesh = scan_mesh.decimate(0.3)

# Compare the scanned mesh to the CAD mesh:
# This will mark points in the scanned mesh that are enclosed by the CAD mesh.
select = scan_mesh.select_enclosed_points(cad_mesh)

# Separate the scanned mesh points based on whether they fall inside or outside the CAD mesh
inside = select.threshold(0.5)            # Points inside the CAD mesh (value 1)
outside = select.threshold(0.5, invert=True)  # Points outside the CAD mesh (value 0)

# Visualize the CAD mesh and the inside/outside parts of the scanned mesh
dargs = dict(show_edges=False)
p = pv.Plotter()
#p.add_mesh(cad_mesh, color="mintcream", opacity=0.8, **dargs)  # CAD mesh as background (semi-transparent)
p.add_mesh(inside, color="gray", opacity=1, **dargs)    # Scanned mesh points inside CAD in green
p.add_mesh(outside, color="Crimson", opacity=0.5, **dargs)  # Scanned mesh points outside CAD in red
p.show()
