import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
"""
This script defines parameters for a double-parallelogram XY flexure stage with serpentine beams, provides helper routines to build and draw rectangular beam segments (cuboids) arranged in sawtooth ‘serpentine’ paths, and then assembles and visualises the complete frame—including outer and intermediate frames, top/bottom X-beams and left/right Y-beams—in 3D using Matplotlib’s Poly3DCollection. The final plot can be saved as an image.
"""

###############################################################################
# 1. PARAMETER CLASS
###############################################################################

class XYFlexureParams:
    """
    Parameters for a basic double-parallelogram XY flexure stage with serpentine beams.
    You can expand these if you want different # of segments for X vs. Y, etc.
    """
    def __init__(
        self,
        outer_size_x=200.0,     # Overall outer frame size in X (mm)
        outer_size_y=150.0,     # Overall outer frame size in Y (mm)
        frame_thickness=5.0,    # Z-thickness of the frame bodies (mm)
        gap=3.0,                # Gap around intermediate/inner frames
        beam_thickness=1.0,     # Z-thickness of each beam (mm)
        beam_width=5.0,         # Cross-section width of each serpentine beam (mm)
        beam_length_x=40.0,     # Horizontal extent of the X-beams (mm)
        beam_length_y=30.0,     # Vertical extent of the Y-beams (mm)
        serpentine_segments_x=4,# Number of segments in each X-beam serpentine
        serpentine_segments_y=4 # Number of segments in each Y-beam serpentine
    ):
        self.outer_size_x = outer_size_x
        self.outer_size_y = outer_size_y
        self.frame_thickness = frame_thickness
        self.gap = gap

        self.beam_thickness = beam_thickness
        self.beam_width = beam_width

        # The “length” the serpentine extends in the primary direction
        self.beam_length_x = beam_length_x
        self.beam_length_y = beam_length_y

        # How many “zigzag” segments for each axis
        self.serpentine_segments_x = serpentine_segments_x
        self.serpentine_segments_y = serpentine_segments_y

###############################################################################
# 2. HELPER: ADD CUBOID
###############################################################################

def add_cuboid(ax, vertices, color='gray', alpha=0.1, edgecolor='k'):
    """
    Given 8 vertices of a rectangular block, add it as a Poly3DCollection.
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

###############################################################################
# 3. BUILD A SERPENTINE BEAM
###############################################################################

def build_serpentine_beam_2d(
    start_x, start_y, end_x, end_y,
    z_center, thickness_z,
    width_xy,
    num_segments,
    orientation='horizontal'
):
    """
    Create a list of cuboids forming a serpentine path from (start_x,start_y)
    to (end_x,end_y) in the XY plane, with thickness in Z around z_center.
    - orientation = 'horizontal' or 'vertical' just to guess how to meander.

    Returns: list of 8-vertex arrays, each a small rectangular block.
    """
    cuboids = []

    # We'll define the total length in the “main” direction
    if orientation == 'horizontal':
        total_length = abs(end_x - start_x)
        sign = 1.0 if (end_x >= start_x) else -1.0
        segment_length = total_length / num_segments
        # We meander in the Y direction by “width_xy,” flipping up/down each time
        current_x = start_x
        current_y = start_y

        # The amplitude of each vertical loop. We can do half up, half down
        # or something. We'll do a simple approach: each segment is horizontal,
        # then the next segment is vertical, etc. But let's keep it simpler:
        # We'll define each sub-segment as a small horizontal rectangle, but
        # it shifts up or down in Y for the serpentine. We'll just do a sawtooth.

        # For i in range(num_segments):
        #   if i is even => y is from start_y to start_y + width_xy
        #   if i is odd  => y is from start_y + width_xy to start_y

        # This means the entire “beam” will wave up and down by beam_width. 
        # Adjust logic to taste.

        direction_up = True
        for i in range(num_segments):
            x1 = current_x
            x2 = current_x + sign*segment_length
            if direction_up:
                # “up” meaning the rectangle extends upward in Y by width_xy
                y1 = start_y
                y2 = start_y + width_xy
            else:
                y1 = start_y - width_xy
                y2 = start_y
            # Build a cuboid from (x1,y1) to (x2,y2) in XY, Z around z_center
            # Thickness in Z = thickness_z
            z1 = z_center - thickness_z/2
            z2 = z_center + thickness_z/2

            v = np.array([
                [x1, y1, z1],
                [x2, y1, z1],
                [x2, y2, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
            ])
            cuboids.append(v)

            # update current_x, direction
            current_x = x2
            direction_up = not direction_up

    else:
        # orientation == 'vertical'
        total_length = abs(end_y - start_y)
        sign = 1.0 if (end_y >= start_y) else -1.0
        segment_length = total_length / num_segments
        # Now we meander in X by “width_xy,” flipping left/right each time
        current_x = start_x
        current_y = start_y
        direction_right = True
        for i in range(num_segments):
            y1 = current_y
            y2 = current_y + sign*segment_length
            if direction_right:
                x1 = start_x
                x2 = start_x + width_xy
            else:
                x1 = start_x - width_xy
                x2 = start_x
            z1 = z_center - thickness_z/2
            z2 = z_center + thickness_z/2

            v = np.array([
                [x1, y1, z1],
                [x2, y1, z1],
                [x2, y2, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
            ])
            cuboids.append(v)

            current_y = y2
            direction_right = not direction_right

    return cuboids

###############################################################################
# 4. VISUALIZATION: DOUBLE-PARALLELOGRAM + SERPENTINE BEAMS
###############################################################################

def visualize_double_parallelogram_serpentine(params: XYFlexureParams, save_path=None):
    """
    Draw a double-parallelogram XY stage with serpentine beams
    for top, bottom, left, right beams.
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')

    # Aliases
    ox = params.outer_size_x
    oy = params.outer_size_y
    tF = params.frame_thickness
    gap = params.gap

    tB = params.beam_thickness  # beam thickness in Z
    wB = params.beam_width      # “meander” amplitude in the serpentine
    # We define the main lengths for the intermediate (X) and stage (Y)
    Lx = params.beam_length_x
    Ly = params.beam_length_y

    # 1) Outer frame as a single rectangular “block”
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
    add_cuboid(ax, outer_rect, color='lightgray', alpha=0.1)

    # Inner rectangle that forms the “outer frame hole”
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
    add_cuboid(ax, inner_rect, color='lightgray', alpha=0.1)

    # 2) Intermediate frame
    int_x1 = irx1 + gap
    int_x2 = int_x1 + Lx
    int_y1 = tF + gap
    int_y2 = oy - tF - gap
    intermediate_rect = np.array([
        [int_x1, int_y1, -tF/2],
        [int_x2, int_y1, -tF/2],
        [int_x2, int_y2, -tF/2],
        [int_x1, int_y2, -tF/2],
        [int_x1, int_y1, tF/2],
        [int_x2, int_y1, tF/2],
        [int_x2, int_y2, tF/2],
        [int_x1, int_y2, tF/2],
    ])
    add_cuboid(ax, intermediate_rect, color='orange', alpha=0.6)

    # 3) “Top” X-beam (SERPENTINE)
    #   We define a serpentine that goes from x_beam_x1 -> x_beam_x2 in X,
    #   at some Y range near the top, meandering in Y by wB.
    top_beam_y1 = int_y2      # anchor near top edge of intermediate frame
    top_beam_y2 = top_beam_y1 + (0.5)  # small offset, or 0 if you prefer
    x_beam_x1 = int_x1
    x_beam_x2 = int_x1 + Lx
    # We'll build sub-cuboids
    top_beam_cuboids = build_serpentine_beam_2d(
        start_x=x_beam_x1,
        start_y=top_beam_y1,
        end_x=x_beam_x2,
        end_y=top_beam_y1,   # same Y because it's “horizontal” overall
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_x,
        orientation='horizontal'
    )
    for c in top_beam_cuboids:
        add_cuboid(ax, c, color='cornflowerblue', alpha=0.5)

    # 4) “Bottom” X-beam (SERPENTINE)
    bottom_beam_y2 = int_y1   # anchor near bottom edge
    bottom_beam_y1 = bottom_beam_y2 - 0.5  # small offset if desired
    bot_beam_cuboids = build_serpentine_beam_2d(
        start_x=x_beam_x1,
        start_y=bottom_beam_y2,
        end_x=x_beam_x2,
        end_y=bottom_beam_y2,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_x,
        orientation='horizontal'
    )
    for c in bot_beam_cuboids:
        add_cuboid(ax, c, color='cornflowerblue', alpha=0.5)

    # 5) Inner stage
    st_x1 = int_x1 + gap
    st_x2 = int_x2 - gap
    st_y_c = (int_y1 + int_y2)/2
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

    # 6) “Left” Y-beam (SERPENTINE)
    left_beam_x2 = st_x1      # anchor near left edge
    left_beam_x1 = left_beam_x2 - 0.5
    # We want a vertical serpentine from (y1->y2).
    # Let's define the stage's center in Y is st_y_c, so we'll meander around that
    y_beam_y1 = st_y_c - (st_h/2)
    y_beam_y2 = st_y_c + (st_h/2)
    left_beam_cuboids = build_serpentine_beam_2d(
        start_x=left_beam_x2,
        start_y=y_beam_y1,
        end_x=left_beam_x2,
        end_y=y_beam_y2,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_y,
        orientation='vertical'
    )
    for c in left_beam_cuboids:
        add_cuboid(ax, c, color='red', alpha=0.8)

    # 7) “Right” Y-beam (SERPENTINE)
    right_beam_x1 = st_x2
    right_beam_x2 = right_beam_x1 + 0.5
    right_beam_cuboids = build_serpentine_beam_2d(
        start_x=right_beam_x1,
        start_y=y_beam_y1,
        end_x=right_beam_x1,
        end_y=y_beam_y2,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_y,
        orientation='vertical'
    )
    for c in right_beam_cuboids:
        add_cuboid(ax, c, color='red', alpha=0.8)

    # Final plot setup
    ax.set_title("Double-Parallelogram Flexure with Serpentine Beams", fontsize=14)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)

    ax.set_xlim(0, params.outer_size_x)
    ax.set_ylim(0, params.outer_size_y)
    ax.set_zlim(-params.outer_size_x/4, params.outer_size_x/4)  # for decent view
    ax.view_init(elev=20, azim=-60)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

###############################################################################
# 5. MAIN
###############################################################################

def main():
    # Example usage with some default parameters
    p = XYFlexureParams(
        outer_size_x=200.0,
        outer_size_y=150.0,
        frame_thickness=5.0,
        gap=3.0,
        beam_thickness=1.0,
        beam_width=5.0,
        beam_length_x=50.0,
        beam_length_y=40.0,
        serpentine_segments_x=5,  # number of zigzag segments for top/bottom
        serpentine_segments_y=4   # number of zigzag segments for left/right
    )
    visualize_double_parallelogram_serpentine(p, save_path="serpentine_2D_flexure.png")

if __name__ == "__main__":
    main()
