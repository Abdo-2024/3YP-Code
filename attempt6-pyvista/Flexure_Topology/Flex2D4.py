import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
"""
This script defines functions to calculate stiffness and maximum deflection for a double-parallelogram flexure stage based on beam theory, including safety‐factor checks against yielding. It then formulates an objective combining mass, deflection penalty, and Z-stiffness ratio penalty, and runs an L-BFGS-B optimisation over beam length, width, thickness, and series count. Finally, it visualises the optimised flexure geometry in 2D using Matplotlib patches and reports the optimal parameters and computed properties.
"""

def calculate_flexure_properties(params):
    """
    Calculate properties of a double parallelogram flexure
    
    Parameters:
    - params: List containing [L, w, t, E, n]
        L = beam length (mm)
        w = beam width (mm) 
        t = beam thickness (mm)
        E = Young's modulus (MPa)
        n = number of flexure units in series
    
    Returns:
    - Dictionary of properties including stiffness and maximum deflection
    """
    L, w, t, E, n = params
    
    # Moment of inertia for rectangular cross-section
    I = (w * t**3) / 12  # mm^4
    
    # Stiffness calculation for double parallelogram (for one direction)
    # Formula: k = 2*E*I/L^3 for a single beam, divided by n for series arrangement
    k_x = 2 * E * I / (L**3) / n  # N/mm
    
    # Safety factor for maximum stress
    yield_strength = 500  # MPa (typical for spring steel)
    safety_factor = 2.0
    
    # Maximum allowable stress
    max_allowable_stress = yield_strength / safety_factor  # MPa
    
    # Maximum deflection before yielding
    # For a cantilever beam: δ_max = (L^2 * σ_max) / (3 * E * t)
    max_deflection_x = (L**2 * max_allowable_stress) / (3 * E * (t/2)) * n  # mm
    
    # Z-axis stiffness (should be much higher)
    k_z = 2 * E * w * t / L * n  # N/mm
    
    return {
        'stiffness_x': k_x,
        'stiffness_z': k_z,
        'max_deflection_x': max_deflection_x,
        'stiffness_ratio': k_z / k_x,
        'mass': 2 * L * w * t * 8e-6 * n,  # kg, assuming density of ~8000 kg/m^3
    }

def objective_function(params):
    """
    Objective function to minimize: a combination of mass and stiffness
    while ensuring required deflection is met
    """
    properties = calculate_flexure_properties(params)
    
    # Penalty for not meeting the deflection requirement of 38mm
    deflection_penalty = max(0, 38 - properties['max_deflection_x'])**2 * 1000
    
    # Penalty for low Z-stiffness ratio (want at least 1000x stiffer in Z)
    z_stiffness_penalty = max(0, 1000 - properties['stiffness_ratio'])**2 * 0.01
    
    # We want to minimize mass while meeting constraints
    return properties['mass'] + deflection_penalty + z_stiffness_penalty

def optimize_flexure(initial_guess, bounds):
    """
    Find optimal flexure parameters using numerical optimization
    """
    result = minimize(
        objective_function,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        return result.x
    else:
        raise ValueError(f"Optimization failed: {result.message}")

def plot_flexure(params):
    """
    Create a simple visualization of the double parallelogram flexure
    """
    L, w, t, E, n = params
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the fixed base
    ax.add_patch(plt.Rectangle((-w/2, -L-t/2), w, L, color='gray', alpha=0.7))
    
    # Draw the flexure beams and connecting blocks for n units
    y_offset = 0
    for i in range(n):
        # First set of flexure beams
        ax.add_patch(plt.Rectangle((-w/2, y_offset), t/2, L, color='blue', alpha=0.7))
        ax.add_patch(plt.Rectangle((w/2-t/2, y_offset), t/2, L, color='blue', alpha=0.7))
        
        # Connecting block
        ax.add_patch(plt.Rectangle((-w/2, y_offset+L), w, t, color='darkgray', alpha=0.7))
        
        # Second set of flexure beams
        ax.add_patch(plt.Rectangle((-w/2, y_offset+L+t), t/2, L, color='blue', alpha=0.7))
        ax.add_patch(plt.Rectangle((w/2-t/2, y_offset+L+t), t/2, L, color='blue', alpha=0.7))
        
        # End platform (if last unit)
        if i == n-1:
            ax.add_patch(plt.Rectangle((-w/2, y_offset+2*L+t), w, t, color='darkgray', alpha=0.7))
        else:
            # Connecting block between units
            ax.add_patch(plt.Rectangle((-w/2, y_offset+2*L+t), w, t*2, color='darkgray', alpha=0.7))
            y_offset += 2*L + 3*t
    
    # Set plot limits and labels
    total_height = n * (2*L + 3*t)
    ax.set_xlim(-w, w)
    ax.set_ylim(-L-t, total_height + t)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Length (mm)')
    ax.set_title('Double Parallelogram Flexure')
    plt.grid(True)
    
    return fig

# Main code
if __name__ == "__main__":
    # Initial parameters: [L, w, t, E, n]
    # L: beam length (mm)
    # w: beam width (mm)
    # t: beam thickness (mm)
    # E: Young's modulus (MPa) - using value for spring steel
    # n: number of flexure units in series
    initial_guess = [50.0, 20.0, 0.5, 200000, 2]
    
    # Parameter bounds
    bounds = [
        (20.0, 200.0),    # L: 20mm to 200mm
        (10.0, 50.0),     # w: 10mm to 50mm
        (0.1, 2.0),       # t: 0.1mm to 2mm
        (200000, 200000), # E: fixed for spring steel
        (1, 5)            # n: 1 to 5 units in series
    ]
    
    # Run optimization
    optimal_params = optimize_flexure(initial_guess, bounds)
    
    # Round parameters for practical manufacturing
    L, w, t, E, n = optimal_params
    n = round(n)  # Must be an integer
    t = round(t * 10) / 10  # Round to nearest 0.1mm
    optimal_params = [L, w, t, E, n]
    
    # Calculate and display results
    properties = calculate_flexure_properties(optimal_params)
    
    print(f"Optimal Double Parallelogram Flexure Parameters:")
    print(f"Beam length (L): {L:.2f} mm")
    print(f"Beam width (w): {w:.2f} mm")
    print(f"Beam thickness (t): {t:.2f} mm")
    print(f"Number of units (n): {int(n)}")
    print("\nFlexure Properties:")
    print(f"X-axis stiffness: {properties['stiffness_x']:.2f} N/mm")
    print(f"Z-axis stiffness: {properties['stiffness_z']:.2f} N/mm") 
    print(f"Z/X stiffness ratio: {properties['stiffness_ratio']:.2f}")
    print(f"Maximum X deflection: {properties['max_deflection_x']:.2f} mm")
    print(f"Total mass: {properties['mass']*1000:.2f} g")
    
    # Plot the flexure
    fig = plot_flexure(optimal_params)
    plt.show()
