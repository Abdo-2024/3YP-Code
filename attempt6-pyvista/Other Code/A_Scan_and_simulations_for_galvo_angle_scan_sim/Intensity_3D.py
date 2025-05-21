import numpy as np
import pyvista as pv
import matplotlib.cm as cm
"""
This script uses PyVista to create a grid of truncated‐cone “needles” with chamfered bases on a rectangular substrate. It defines intensity‐based scalar colouring on each frustum based on radial distance, builds each chamfered cone by merging two frustum segments, and arranges a 10×10 array of these cones on a thin base block. The scene is rendered with a grayscale colormap, axes, and bounds.
"""

def circle_intensity(r, r_inner, r_outer):
    """
    Piecewise intensity function:
      - 0 for r <= r_inner,
      - Linear ramp for r between r_inner and r_outer,
      - 1 for r >= r_outer.
    """
    I = np.ones_like(r)
    I[r <= r_inner] = 0.0
    between = (r > r_inner) & (r < r_outer)
    I[between] = (r[between] - r_inner) / (r_outer - r_inner)
    return I

def make_truncated_cone(bottom_radius, top_radius, height, resolution=60, r_inner=None, r_outer=None):
    """
    Create a truncated cone (frustum) along +Z.
    Computes a scalar field on vertices based on radial distance (x,y) using the
    circle_intensity function. We then scale the scalars so that the darkest is not pure black.
    """
    if r_inner is None:
        r_inner = top_radius
    if r_outer is None:
        r_outer = bottom_radius

    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    
    # Bottom circle (z=0)
    xb = bottom_radius * np.cos(angles)
    yb = bottom_radius * np.sin(angles)
    zb = np.zeros_like(angles)
    
    # Top circle (z=height)
    xt = top_radius * np.cos(angles)
    yt = top_radius * np.sin(angles)
    zt = np.full_like(angles, height)
    
    bottom_circle = np.column_stack((xb, yb, zb))
    top_circle = np.column_stack((xt, yt, zt))
    vertices = np.vstack((bottom_circle, top_circle))
    
    # Compute intensity based solely on the radial distance from the local center.
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    scalars = circle_intensity(r, r_inner, r_outer)
    # Scale the scalars so that they range from 0.3 (dark gray) to 1 (white) instead of 0 to 1.
    scalars = 0 + 1 * scalars
    
    polydata = pv.PolyData()
    polydata.points = vertices
    polydata["scalars"] = scalars

    # Build faces: side triangles and polygonal top and bottom.
    faces = []
    n_bottom = resolution
    for i in range(resolution):
        i_next = (i + 1) % resolution
        faces.append([3, i, i_next, n_bottom + i])
        faces.append([3, n_bottom + i, i_next, n_bottom + i_next])
    faces.append([resolution] + list(range(n_bottom)))
    faces.append([resolution] + list(range(n_bottom, n_bottom + resolution)))
    
    faces_flat = []
    for f in faces:
        faces_flat.extend(f)
    polydata.faces = np.array(faces_flat, dtype=np.int64)
    
    return polydata

def make_cone_with_chamfer(bottom_radius, top_radius, needle_height,
                           chamfer_height=30.0, chamfer_delta=15.0, resolution=100):
    """
    Create a cone with a chamfered base:
      1. A chamfer piece from z=0 to chamfer_height:
         bottom radius = bottom_radius, top radius = bottom_radius - chamfer_delta.
      2. A main cone from z=chamfer_height to needle_height:
         bottom radius = bottom_radius - chamfer_delta, top radius = top_radius.
    The two pieces are merged together.
    """
    # Chamfer piece: intensity mapping with r_inner = bottom_radius - chamfer_delta, r_outer = bottom_radius.
    chamfer_piece = make_truncated_cone(bottom_radius, bottom_radius - chamfer_delta,
                                        chamfer_height, resolution=resolution,
                                        r_inner=bottom_radius - chamfer_delta, r_outer=bottom_radius)
    
    # Main cone piece: intensity mapping with r_inner = top_radius, r_outer = bottom_radius - chamfer_delta.
    main_piece = make_truncated_cone(bottom_radius - chamfer_delta, top_radius,
                                     needle_height - chamfer_height, resolution=resolution,
                                     r_inner=top_radius, r_outer=bottom_radius - chamfer_delta)
    # Translate the main piece upward so its bottom aligns with the top of the chamfer.
    main_piece = main_piece.translate((0, 0, chamfer_height), inplace=False)
    
    # Merge the two pieces.
    combined = chamfer_piece.merge(main_piece)
    return combined

def main():
    # Base block: 3.6 mm x 3.6 mm x 50 µm.
    width_um = 3600.0
    height_um = 3600.0
    thickness_um = 550.0

    base_center = (0, 0, thickness_um / 2.0)
    base = pv.Cube(center=base_center,
                   x_length=width_um, y_length=height_um, z_length=thickness_um)

    # Create grid positions (10×10) with a 200 µm margin.
    grid_size = 10
    margin = 200.0
    x_positions = np.linspace(-width_um/2 + margin, width_um/2 - margin, grid_size)
    y_positions = np.linspace(-height_um/2 + margin, height_um/2 - margin, grid_size)
    centers = [(x, y) for x in x_positions for y in y_positions]

    # Cone parameters (in µm).
    needle_height = 450.0
    bottom_radius = 109/2  # outer radius from 2D intensity (white edge).
    top_radius = 15/2      # inner radius from 2D intensity (black center).
    chamfer_height = 50.0
    chamfer_delta = 20.0

    plotter = pv.Plotter()
    # Add the base without edges.
    plotter.add_mesh(base, color="lightgray", show_edges=False)

    # Create and add cones with chamfer, colored by the computed scalars using a gray colormap.
    for (cx, cy) in centers:
        cone = make_cone_with_chamfer(bottom_radius, top_radius, needle_height,
                                      chamfer_height, chamfer_delta, resolution=60)
        # Translate the cone so its bottom sits on top of the base.
        cone = cone.translate((cx, cy, thickness_um), inplace=False)
        plotter.add_mesh(cone, scalars="scalars", cmap="gray", show_edges=False)
    
    plotter.add_axes(interactive=True)
    plotter.show_bounds(grid="front")
    plotter.reset_camera()
    plotter.view_isometric()
    plotter.show()

if __name__ == "__main__":
    main()
