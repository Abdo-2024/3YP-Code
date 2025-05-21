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
