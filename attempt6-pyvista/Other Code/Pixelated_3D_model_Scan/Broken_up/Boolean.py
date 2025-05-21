# ------------------------------------------------------------------------------
# Step 3: Boolean operations with safer checks
# ------------------------------------------------------------------------------

# Boolean.py

def safe_union(base_mesh, sphere):
    """
    Attempt a boolean union of 'sphere' with 'base_mesh'.
    If intersection is present but fails, skip. If no intersection, try anyway.
    """
    result = base_mesh.intersection(sphere)
    # If intersection(...) returned a tuple, take the first part
    if isinstance(result, tuple):
        imesh = result[0]
    else:
        imesh = result

    if imesh is None or imesh.n_points == 0:
        # No real intersection, but we can still union if it doesn't fail
        try:
            return base_mesh.boolean_union(sphere, tolerance=1e-3)
        except Exception as e:
            print("Union failed on a non-intersecting sphere:", e)
            return base_mesh
    else:
        # We do have an intersection
        try:
            return base_mesh.boolean_union(sphere, tolerance=1e-6)
        except Exception as e:
            print("Union failed on an intersecting sphere:", e)
            return base_mesh


def safe_difference(base_mesh, sphere):
    """
    Attempt a boolean difference to subtract 'sphere' from 'base_mesh'.
    If there's no intersection or if the operation fails, skip it.
    """
    result = base_mesh.intersection(sphere)
    if isinstance(result, tuple):
        imesh = result[0]
    else:
        imesh = result

    if imesh is None or imesh.n_points == 0:
        # No intersection => no subtraction
        return base_mesh
    else:
        try:
            return base_mesh.boolean_difference(sphere, tolerance=1e-3)
        except Exception as e:
            print("Difference failed on a sphere:", e)
            return base_mesh
