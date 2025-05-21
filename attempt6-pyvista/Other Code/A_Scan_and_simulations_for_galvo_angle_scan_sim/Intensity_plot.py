import numpy as np
import matplotlib.pyplot as plt
"""
This script generates and visualises a 2D intensity map on a 3.6 mm × 3.6 mm stage by placing a 10×10 grid of concentric circles with smooth gradients between an inner and outer radius. It:
1. Defines a piecewise ‘circle_intensity’ function (zero inside, linear ramp, then one outside).
2. Converts the stage dimensions to a pixel grid at a specified resolution (7 µm per pixel).
3. Computes the intensity for each pixel by taking the minimum across all circles in the grid.
4. Plots the resulting grayscale image with Matplotlib, showing 100 smoothly-edged circles.
"""

def circle_intensity(r, r_inner, r_outer):
    """
    Compute the piecewise intensity based on the distance r from the circle center:
      - 0 for r <= r_inner
      - Linear ramp between r_inner and r_outer
      - 1 for r >= r_outer
    """
    I = np.ones_like(r)  # default: outside outer radius is max intensity (1)
    inside_inner = (r <= r_inner)
    I[inside_inner] = 0.0
    between = (r > r_inner) & (r < r_outer)
    I[between] = (r[between] - r_inner) / (r_outer - r_inner)
    return I

def create_pattern(
    width_mm=3.6,
    height_mm=3.6,
    r_inner_um=7.5,    # 15 µm diameter inner circle -> radius = 7.5 µm
    r_outer_um=54.5,   # 109 µm diameter outer circle -> radius = 54.5 µm
    resolution_um=7.0, # each pixel is 7 µm
    grid_size=10       # 10 x 10 grid gives 100 circles
):
    """
    Creates a 2D intensity map on a 3.6 mm x 3.6 mm stage, placing 'grid_size x grid_size'
    circles. Each circle has an inner and outer radius with a smooth gradient between.
    """
    # Convert dimensions from mm to µm
    width_um = width_mm * 1000.0   # 3600 µm
    height_um = height_mm * 1000.0   # 3600 µm

    # Number of pixels based on the desired resolution (7 µm per pixel)
    nx = int(width_um / resolution_um)
    ny = int(height_um / resolution_um)

    # Create coordinate arrays (µm), centered at (0, 0)
    x_vals = np.linspace(-width_um/2, width_um/2, nx)
    y_vals = np.linspace(-height_um/2, height_um/2, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Initialize the image to max intensity (1)
    image = np.ones((ny, nx))

    # Define circle centers in a 10x10 grid
    # Using a margin to avoid circles going off the stage.
    margin = 200.0  # µm margin from the edge
    x_centers = np.linspace(-width_um/2 + margin, width_um/2 - margin, grid_size)
    y_centers = np.linspace(-height_um/2 + margin, height_um/2 - margin, grid_size)
    circle_centers = [(x, y) for x in x_centers for y in y_centers]

    # For each circle, compute the distance map and the intensity, then merge via minimum.
    for (cx, cy) in circle_centers:
        rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        circle_vals = circle_intensity(rr, r_inner_um, r_outer_um)
        image = np.minimum(image, circle_vals)

    return xx, yy, image

def main():
    # Create the pattern with a resolution of 7 µm per pixel
    xx, yy, image = create_pattern(
        width_mm=3.6,
        height_mm=3.6,
        r_inner_um=7.5,     # inner circle radius (15 µm diameter)
        r_outer_um=54.5,    # outer circle radius (109 µm diameter)
        resolution_um=7.0,  # 7 µm per pixel
        grid_size=10        # 10x10 grid -> 100 circles
    )

    # Plot the resulting intensity map
    plt.figure(figsize=(6,6))
    plt.imshow(
        image,
        extent=[xx[0,0], xx[0,-1], yy[0,0], yy[-1,0]],
        origin='lower',
        cmap='gray'
    )
    plt.colorbar(label='Intensity')
    plt.title("100 Circles with Inner=15 µm, Outer=109 µm (7 µm/px)")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.show()

if __name__ == "__main__":
    main()
