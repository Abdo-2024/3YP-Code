import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
"""
This script defines parameters for an exaggerated double-parallelogram XY flexure stage with serpentine beams, and visualises it in 3D using Matplotlib. It includes:
1. A parameter class to specify frame and beam dimensions and serpentine segment counts.
2. A helper function to add cuboid blocks (frame walls and beam segments) to a 3D plot.
3. A routine to build serpentine beam geometries by subdividing in the primary direction and meandering in the orthogonal direction.
4. An assembly and rendering of the complete flexure: outer frame, inner frame hole, intermediate frame, top/bottom horizontal serpentine beams, inner stage, and left/right vertical serpentine beams, with customizable colours and transparency.
5. A main function to instantiate parameters and save the resulting plot.
"""

###############################################################################
# 1. Parameter Class
###############################################################################

class XYFlexureParams:
    def __init__(
        self,
        outer_size_x=200.0,
        outer_size_y=150.0,
        frame_thickness=2.0,      # Make the frame thinner so beams are visible
        gap=5.0,                  # Larger gap around frames
        beam_thickness=1.0,       # Z-thickness of each beam
        beam_width=15.0,          # SERPENTINE AMPLITUDE - bigger so it's visible
        beam_length_x=80.0,       # Horizontal extent for X-beams
        beam_length_y=50.0,       # Vertical extent for Y-beams
        serpentine_segments_x=6,  # # of zigzag segments for X beams
        serpentine_segments_y=6   # # of zigzag segments for Y beams
    ):
        self.outer_size_x = outer_size_x
        self.outer_size_y = outer_size_y
        self.frame_thickness = frame_thickness
        self.gap = gap

        self.beam_thickness = beam_thickness
        self.beam_width = beam_width

        self.beam_length_x = beam_length_x
        self.beam_length_y = beam_length_y

        self.serpentine_segments_x = serpentine_segments_x
        self.serpentine_segments_y = serpentine_segments_y

###############################################################################
# 2. Utility: Add Cuboid
###############################################################################

def add_cuboid(ax, vertices, color='gray', alpha=1.0, edgecolor='k'):
    faces_idx = [
        [0,1,2,3],
        [4,5,6,7],
        [0,1,5,4],
        [2,3,7,6],
        [0,3,7,4],
        [1,2,6,5],
    ]
    for f in faces_idx:
        face_coords = [vertices[i] for i in f]
        poly = Poly3DCollection([face_coords], facecolors=color, edgecolors=edgecolor, alpha=alpha)
        ax.add_collection3d(poly)

###############################################################################
# 3. Build a Serpentine Beam in 2D
###############################################################################

def build_serpentine_beam_2d(
    start_x, start_y,
    end_x, end_y,
    z_center, thickness_z,
    width_xy,
    num_segments,
    orientation='horizontal'
):
    """
    Create a list of cuboids forming a serpentine in the XY plane.
    For 'horizontal', we subdivide in X and meander in Y by ± width_xy.
    For 'vertical', we subdivide in Y and meander in X by ± width_xy.
    """
    cuboids = []

    if orientation == 'horizontal':
        total_length = abs(end_x - start_x)
        if total_length < 1e-9:
            return cuboids  # no length => no beam
        seg_len = total_length / num_segments
        direction_up = True
        current_x = start_x

        for _ in range(num_segments):
            x1 = current_x
            x2 = current_x + seg_len
            if direction_up:
                # meander up
                y1 = start_y
                y2 = start_y + width_xy
            else:
                # meander down
                y1 = start_y - width_xy
                y2 = start_y
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

            current_x = x2
            direction_up = not direction_up

    else:  # orientation == 'vertical'
        total_length = abs(end_y - start_y)
        if total_length < 1e-9:
            return cuboids
        seg_len = total_length / num_segments
        direction_right = True
        current_y = start_y

        for _ in range(num_segments):
            y1 = current_y
            y2 = current_y + seg_len
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
# 4. Visualize Double-Parallelogram w/ Serpentine Beams
###############################################################################

def visualize_double_parallelogram_serpentine(params: XYFlexureParams, save_path=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')

    ox = params.outer_size_x
    oy = params.outer_size_y
    tF = params.frame_thickness
    gap = params.gap

    tB = params.beam_thickness
    wB = params.beam_width

    Lx = params.beam_length_x
    Ly = params.beam_length_y

    # 1) Outer rectangular "frame"
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

    # 2) "Hole" inside the outer frame
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

    # 3) Intermediate frame
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

    # 4) TOP X-BEAM (serpentine)
    top_beam_y = int_y2 + 2  # +2 mm offset so it's visibly above the frame
    # we meander from x_beam_x1 -> x_beam_x2 horizontally
    x_beam_x1 = int_x1
    x_beam_x2 = int_x2
    top_beam_cuboids = build_serpentine_beam_2d(
        start_x=x_beam_x1,
        start_y=top_beam_y,
        end_x=x_beam_x2,
        end_y=top_beam_y,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,               # amplitude of up/down meander
        num_segments=params.serpentine_segments_x,
        orientation='horizontal'
    )
    for c in top_beam_cuboids:
        add_cuboid(ax, c, color='cornflowerblue', alpha=0.8)

    # 5) BOTTOM X-BEAM (serpentine)
    bottom_beam_y = int_y1 - 2
    bot_beam_cuboids = build_serpentine_beam_2d(
        start_x=x_beam_x1,
        start_y=bottom_beam_y,
        end_x=x_beam_x2,
        end_y=bottom_beam_y,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_x,
        orientation='horizontal'
    )
    for c in bot_beam_cuboids:
        add_cuboid(ax, c, color='cornflowerblue', alpha=0.8)

    # 6) Inner stage
    st_x1 = int_x1 + gap
    st_x2 = int_x2 - gap
    st_y_c = (int_y1 + int_y2)/2
    st_h = Ly
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

    # 7) LEFT Y-BEAM (serpentine)
    left_beam_x = st_x1 - 2
    y_beam_y1 = st_y1
    y_beam_y2 = st_y2
    left_beam_cuboids = build_serpentine_beam_2d(
        start_x=left_beam_x,
        start_y=y_beam_y1,
        end_x=left_beam_x,
        end_y=y_beam_y2,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_y,
        orientation='vertical'
    )
    for c in left_beam_cuboids:
        add_cuboid(ax, c, color='red', alpha=0.8)

    # 8) RIGHT Y-BEAM (serpentine)
    right_beam_x = st_x2 + 2
    right_beam_cuboids = build_serpentine_beam_2d(
        start_x=right_beam_x,
        start_y=y_beam_y1,
        end_x=right_beam_x,
        end_y=y_beam_y2,
        z_center=0.0,
        thickness_z=tB,
        width_xy=wB,
        num_segments=params.serpentine_segments_y,
        orientation='vertical'
    )
    for c in right_beam_cuboids:
        add_cuboid(ax, c, color='red', alpha=0.8)

    ax.set_title("Double-Parallelogram with Serpentine Beams (Exaggerated)", fontsize=14)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)

    ax.set_xlim(0, ox)
    ax.set_ylim(0, oy)
    ax.set_zlim(-ox/5, ox/5)  # so we can see
    ax.view_init(elev=20, azim=-60)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

###############################################################################
# 5. Main
###############################################################################

def main():
    p = XYFlexureParams(
        outer_size_x=200.0,
        outer_size_y=150.0,
        frame_thickness=2.0,
        gap=5.0,
        beam_thickness=1.0,
        beam_width=15.0,        # serpentine amplitude
        beam_length_x=80.0,
        beam_length_y=50.0,
        serpentine_segments_x=6,
        serpentine_segments_y=6
    )
    visualize_double_parallelogram_serpentine(p, save_path="serpentine_demo.png")

if __name__ == "__main__":
    main()
