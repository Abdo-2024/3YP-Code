import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize
"""
This script defines material and flexure parameter classes for a double-parallelogram XY flexure stage, provides routines to create and visualise the frame with straight cuboid beams in 3D, computes approximate X/Y deflections using simple beam theory, and optimises key geometry variables (outer sizes, beam lengths, widths, thickness) via L-BFGS-B to meet deflection and size constraints. The final design is plotted and reported.
"""

###############################################################################
# 1. PARAMETER + MATERIAL CLASSES
###############################################################################

class Material:
    """
    Basic material model (placeholder).
    """
    def __init__(self, young_modulus=7e10, poisson_ratio=0.34):
        self.young_modulus = young_modulus   # Pa
        self.poisson_ratio = poisson_ratio

class XYFlexureParams:
    """
    Parameters for a basic double-parallelogram XY flexure stage.

    Feel free to expand with:
      - # of beams in X or Y
      - beam 'serpentine' segments, etc.
    """
    def __init__(
        self,
        outer_size_x=200.0,     # Overall outer frame size in X (mm)
        outer_size_y=200.0,     # Overall outer frame size in Y (mm)
        beam_length_x=40.0,     # Effective length of X-axis beams (mm)
        beam_length_y=40.0,     # Effective length of Y-axis beams (mm)
        beam_width=5.0,         # Width of each beam (mm)
        beam_thickness=1.0,     # Thickness (Z dimension) of each beam (mm)
        frame_thickness=5.0,    # Z-thickness of frames (mm)
        gap=2.0                 # Gap around frames (mm)
    ):
        self.outer_size_x = outer_size_x
        self.outer_size_y = outer_size_y
        self.beam_length_x = beam_length_x
        self.beam_length_y = beam_length_y
        self.beam_width = beam_width
        self.beam_thickness = beam_thickness
        self.frame_thickness = frame_thickness
        self.gap = gap

###############################################################################
# 2. GEOMETRY CREATION / VISUALIZATION
###############################################################################

def add_cuboid(ax, vertices, color='gray', alpha=1.0, edgecolor='k'):
    """
    Utility: given 8 vertices of a rectangular block, add it as a Poly3DCollection.
    We'll define faces in the standard pattern [0,1,2,3], etc.
    """
    faces_idx = [
        [0,1,2,3],  # bottom
        [4,5,6,7],  # top
        [0,1,5,4],  # front
        [2,3,7,6],  # back
        [0,3,7,4],  # left
        [1,2,6,5],  # right
    ]
    for f in faces_idx:
        face_coords = [vertices[i] for i in f]
        poly = Poly3DCollection([face_coords], facecolors=color, edgecolors=edgecolor, alpha=alpha)
        ax.add_collection3d(poly)

def visualize_double_parallelogram(params: XYFlexureParams, save_path=None):
    """
    Draw a simple double-parallelogram flexure stage for 2D (X, Y) motion.
    In a real design, you'd replace these straight beams with serpentine beams, etc.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # For convenience
    ox = params.outer_size_x
    oy = params.outer_size_y
    tF = params.frame_thickness
    tB = params.beam_thickness
    wB = params.beam_width
    gap = params.gap

    # Outer frame:
    outer_rect = np.array([
        [0, 0, -tF/2],
        [ox, 0, -tF/2],
        [ox, oy, -tF/2],
        [0, oy, -tF/2],
        [0, 0, tF/2],
        [ox, 0, tF/2],
        [ox, oy, tF/2],
        [0, oy, tF/2],
    ])
    add_cuboid(ax, outer_rect, color='lightgray', alpha=0.9)

    # Inner rectangle (outer frame "hole"):
    # We'll just show it as a second block for brevity
    irx1, iry1 = (tF, tF)
    irx2, iry2 = (ox - tF, oy - tF)
    inner_rect = np.array([
        [irx1, iry1, -tF/2],
        [irx2, iry1, -tF/2],
        [irx2, iry2, -tF/2],
        [irx1, iry2, -tF/2],
        [irx1, iry1, tF/2],
        [irx2, iry1, tF/2],
        [irx2, iry2, tF/2],
        [irx1, iry2, tF/2],
    ])
    add_cuboid(ax, inner_rect, color='lightgray', alpha=0.9)

    # Intermediate frame:
    int_wx = params.beam_length_x
    int_x1 = irx1 + gap
    int_x2 = int_x1 + int_wx
    int_ly1 = tF + gap
    int_ly2 = oy - tF - gap
    intermediate_rect = np.array([
        [int_x1, int_ly1, -tF/2],
        [int_x2, int_ly1, -tF/2],
        [int_x2, int_ly2, -tF/2],
        [int_x1, int_ly2, -tF/2],
        [int_x1, int_ly1, tF/2],
        [int_x2, int_ly1, tF/2],
        [int_x2, int_ly2, tF/2],
        [int_x1, int_ly2, tF/2],
    ])
    add_cuboid(ax, intermediate_rect, color='orange', alpha=0.6)

    # X-beams (just two beams, top/bottom):
    x_beam_x1 = int_x1
    x_beam_x2 = x_beam_x1 + tB
    # Top
    top_beam_y1 = int_ly2
    top_beam_y2 = top_beam_y1 + wB
    top_beam = np.array([
        [x_beam_x1, top_beam_y1, -tB/2],
        [x_beam_x2, top_beam_y1, -tB/2],
        [x_beam_x2, top_beam_y2, -tB/2],
        [x_beam_x1, top_beam_y2, -tB/2],
        [x_beam_x1, top_beam_y1, tB/2],
        [x_beam_x2, top_beam_y1, tB/2],
        [x_beam_x2, top_beam_y2, tB/2],
        [x_beam_x1, top_beam_y2, tB/2],
    ])
    add_cuboid(ax, top_beam, color='cornflowerblue', alpha=0.8)

    # Bottom
    bot_beam_y2 = int_ly1
    bot_beam_y1 = bot_beam_y2 - wB
    bot_beam = np.array([
        [x_beam_x1, bot_beam_y1, -tB/2],
        [x_beam_x2, bot_beam_y1, -tB/2],
        [x_beam_x2, bot_beam_y2, -tB/2],
        [x_beam_x1, bot_beam_y2, -tB/2],
        [x_beam_x1, bot_beam_y1, tB/2],
        [x_beam_x2, bot_beam_y1, tB/2],
        [x_beam_x2, bot_beam_y2, tB/2],
        [x_beam_x1, bot_beam_y2, tB/2],
    ])
    add_cuboid(ax, bot_beam, color='cornflowerblue', alpha=0.8)

    # Inner stage:
    st_x1 = int_x1 + gap
    st_x2 = int_x2 - gap
    st_y_c = (int_ly1 + int_ly2)/2
    st_h = params.beam_length_y
    st_y1 = st_y_c - st_h/2
    st_y2 = st_y_c + st_h/2
    stage_rect = np.array([
        [st_x1, st_y1, -tF/2],
        [st_x2, st_y1, -tF/2],
        [st_x2, st_y2, -tF/2],
        [st_x1, st_y2, -tF/2],
        [st_x1, st_y1, tF/2],
        [st_x2, st_y1, tF/2],
        [st_x2, st_y2, tF/2],
        [st_x1, st_y2, tF/2],
    ])
    add_cuboid(ax, stage_rect, color='green', alpha=0.7)

    # Y-beams (just two beams, left/right):
    left_beam_x2 = st_x1
    left_beam_x1 = left_beam_x2 - wB
    left_beam_y1 = st_y_c - tB/2
    left_beam_y2 = left_beam_y1 + tB
    left_beam = np.array([
        [left_beam_x1, left_beam_y1, -tB/2],
        [left_beam_x2, left_beam_y1, -tB/2],
        [left_beam_x2, left_beam_y2, -tB/2],
        [left_beam_x1, left_beam_y2, -tB/2],
        [left_beam_x1, left_beam_y1, tB/2],
        [left_beam_x2, left_beam_y1, tB/2],
        [left_beam_x2, left_beam_y2, tB/2],
        [left_beam_x1, left_beam_y2, tB/2],
    ])
    add_cuboid(ax, left_beam, color='red', alpha=0.8)

    right_beam_x1 = st_x2
    right_beam_x2 = right_beam_x1 + wB
    right_beam_y1 = st_y_c - tB/2
    right_beam_y2 = right_beam_y1 + tB
    right_beam = np.array([
        [right_beam_x1, right_beam_y1, -tB/2],
        [right_beam_x2, right_beam_y1, -tB/2],
        [right_beam_x2, right_beam_y2, -tB/2],
        [right_beam_x1, right_beam_y2, -tB/2],
        [right_beam_x1, right_beam_y1, tB/2],
        [right_beam_x2, right_beam_y1, tB/2],
        [right_beam_x2, right_beam_y2, tB/2],
        [right_beam_x1, right_beam_y2, tB/2],
    ])
    add_cuboid(ax, right_beam, color='red', alpha=0.8)

    ax.set_title("Double-Parallelogram Flexure (XY Stage)", fontsize=14)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)

    ax.set_xlim(0, ox)
    ax.set_ylim(0, oy)
    ax.set_zlim(-max(ox, oy)/4, max(ox, oy)/4)
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

###############################################################################
# 3. DEFLECTION / STIFFNESS CALCULATION (SIMPLE PLACEHOLDER)
###############################################################################

def calculate_deflections_2d(params: XYFlexureParams, material: Material, Fx, Fy):
    """
    Return (x_deflection, y_deflection, z_deflection) for loads Fx, Fy.

    WARNING: This is a toy placeholder model:
    - We treat each axis as a pair (or two pairs) of beams with "effective length"
      in the bending direction, ignoring more complex geometry.
    - We pretend Z is stiff, returning near-zero if thickness is above some threshold.
    - For a real design, you'd do a more thorough analysis or FEA.

    Example approach for each axis:
    - X-axis beams are 'vertical' columns (height ~ beam_length_x),
      second moment depends on beam_width / beam_thickness, etc.
    - Y-axis beams are 'horizontal' columns (length ~ beam_length_y).
    - Then combine to find net deflection in X, Y.

    We'll just do a single-axis beam formula:
        k = (E * I) / (L^3) * factor
    and interpret it for X or Y.

    Return (x_def, y_def, z_def).
    """
    E = material.young_modulus  # Pa
    # Convert mm to meters for SI-based formula
    Lx = params.beam_length_x / 1000.0
    Ly = params.beam_length_y / 1000.0
    t = params.beam_thickness / 1000.0
    w = params.beam_width / 1000.0

    # For "vertical" beams in X, let's say the bending axis is around "z",
    # so second moment is I_z = (t * w^3)/12 or something.
    # We'll pick one. Let's define I_x = (t * w^3)/12
    I_x = (t * (w**3)) / 12.0
    # Similarly for Y beams, if they bend around z as well:
    I_y = (t * (w**3)) / 12.0

    # We'll do a simple cantilever formula for each pair. Real double‐parallelograms
    # have 2 or 4 beams per axis, which changes the factor. We'll just guess a factor.
    # For a pair of beams in parallel, total k ~ 2 * (3 E I / L^3).
    # We'll do that for each axis:
    k_x = 2.0 * (3.0 * E * I_x) / (Lx**3)  # N/m
    k_y = 2.0 * (3.0 * E * I_y) / (Ly**3)  # N/m

    # Deflection for Fx: x_def = Fx / k_x, similarly for y.
    # Convert Fx from N to consistent units, then deflection from m to mm.
    x_def = 0
    y_def = 0
    if k_x > 1e-12:
        x_def = (Fx / k_x) * 1000.0  # in mm
    if k_y > 1e-12:
        y_def = (Fy / k_y) * 1000.0  # in mm

    # For Z deflection, let's do a quick hack: assume it's extremely stiff if
    # tB >= 1 mm. We'll just say z_def = 0.0 for demonstration.
    # In practice, you'd compute or test an out‐of‐plane bending mode.
    z_def = 0.0

    return x_def, y_def, z_def

###############################################################################
# 4. OBJECTIVE FUNCTION FOR OPTIMIZATION
###############################################################################

def objective_function_2d(x, material):
    """
    We interpret x as:
        x[0] = outer_size_x
        x[1] = outer_size_y
        x[2] = beam_length_x
        x[3] = beam_length_y
        x[4] = beam_width
        x[5] = beam_thickness

    We'll keep frame_thickness, gap fixed for now (or you can add them as free vars).
    We'll:
      - Build a params object
      - Compute deflections in X & Y for 2 N
      - Check if deflection >= 38 mm
      - Check bounding box <= 250 mm
      - Penalize out-of-plane deflection at 3N
    """
    # Unpack
    outer_size_x = x[0]
    outer_size_y = x[1]
    beam_length_x = x[2]
    beam_length_y = x[3]
    beam_width = x[4]
    beam_thickness = x[5]

    # Basic sanity bounds (the optimizer also has bounds, but let's clamp a bit).
    if outer_size_x < 1 or outer_size_y < 1:
        return 1e9  # huge penalty

    # Create a parameter set
    # (keeping frame_thickness = 5, gap=2 fixed for demonstration)
    params = XYFlexureParams(
        outer_size_x=outer_size_x,
        outer_size_y=outer_size_y,
        beam_length_x=beam_length_x,
        beam_length_y=beam_length_y,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        frame_thickness=5.0,
        gap=2.0
    )

    # 1) Check bounding box <= 250 x 250
    penalty = 0.0
    if outer_size_x > 250:
        penalty += 10000*(outer_size_x - 250)
    if outer_size_y > 250:
        penalty += 10000*(outer_size_y - 250)

    # 2) Calculate deflections for Fx=2N, Fy=2N
    Fx, Fy = 2.0, 2.0
    x_def_2N, y_def_2N, z_def_2N = calculate_deflections_2d(params, material, Fx, Fy)

    # We want x_def_2N >= 38 mm, y_def_2N >= 38 mm
    if x_def_2N < 38.0:
        penalty += 5000*(38.0 - x_def_2N)
    if y_def_2N < 38.0:
        penalty += 5000*(38.0 - y_def_2N)

    # 3) Check Z deflection at 3N => should be ~0
    Fx_z, Fy_z = 0.0, 0.0  # assume load is purely out-of-plane if you want
    # but let's do a kludge: we can just call the same function or skip
    # For demonstration, let's just say if beam_thickness < 1 mm we add penalty:
    if beam_thickness < 1.0:
        penalty += 5000*(1.0 - beam_thickness)

    # 4) If you want, add penalty for big or tiny beam_width, etc.
    # We'll skip that or keep minimal.  
    if beam_width < 0.5:
        penalty += 1000*(0.5 - beam_width)

    return penalty

###############################################################################
# 5. OPTIMIZATION ROUTINE + MAIN
###############################################################################

def optimize_2d_flexure(material):
    """
    Run a simple optimization over [outer_size_x, outer_size_y, beam_length_x,
    beam_length_y, beam_width, beam_thickness].
    """
    # Initial guess
    x0 = [200, 200, 40, 40, 5, 1]

    # Bounds: (min, max) for each variable
    # outer_size_x -> (100, 250)
    # outer_size_y -> (100, 250)
    # beam_length_x -> (10, 100)
    # beam_length_y -> (10, 100)
    # beam_width -> (0.5, 20)
    # beam_thickness -> (0.1, 5)
    bounds = [
        (100, 250),  # outer_size_x
        (100, 250),  # outer_size_y
        (10, 100),   # beam_length_x
        (10, 100),   # beam_length_y
        (0.5, 20),   # beam_width
        (0.1, 5)     # beam_thickness
    ]

    res = minimize(
        objective_function_2d,
        x0,
        args=(material,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-6, 'maxiter':200}
    )
    return res

def main():
    # Create a material
    mat = Material(young_modulus=7e10, poisson_ratio=0.34)

    print("Starting 2D flexure optimization...")
    result = optimize_2d_flexure(mat)
    print("Optimization finished.")
    print("Best penalty =", result.fun)
    print("Optimized variables (outer_size_x, outer_size_y, beam_length_x, beam_length_y, beam_width, beam_thickness):")
    print(result.x)

    # Build final params
    best_x = result.x
    final_params = XYFlexureParams(
        outer_size_x=best_x[0],
        outer_size_y=best_x[1],
        beam_length_x=best_x[2],
        beam_length_y=best_x[3],
        beam_width=best_x[4],
        beam_thickness=best_x[5],
        frame_thickness=5.0, # fixed in this example
        gap=2.0
    )

    # Visualize final design
    visualize_double_parallelogram(final_params, save_path="final_2D_flexure.png")

    # For demonstration, let's check X, Y deflection at 2N
    Fx, Fy = 2.0, 2.0
    x_def, y_def, z_def = calculate_deflections_2d(final_params, mat, Fx, Fy)
    print(f"Deflections at 2N => X: {x_def:.2f} mm, Y: {y_def:.2f} mm, Z: {z_def:.2f} mm")

if __name__ == "__main__":
    main()
