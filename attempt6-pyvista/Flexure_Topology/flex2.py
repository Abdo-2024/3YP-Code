import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
"""
This script defines material and flexure parameters, calculates beam deflection and stiffness using cantilever theory for different configurations, and optimises flexure geometry (length, width, thickness, number of beams) to meet target deflection and size constraints via L-BFGS-B. It then visualises the serpentine flexure geometry in 3D, analyses deflection vs. force performance, checks design requirements, and outputs plots and a performance report.
"""

# Material properties for aluminum
class Material:
    def __init__(self):
        self.young_modulus = 7e10  # Pa
        self.poisson_ratio = 0.34
        self.density = 2700  # kg/m^3

# Flexure beam parameters
class FlexureParameters:
    def __init__(self, length, width, thickness, num_beams=1, configuration='parallel'):
        self.length = length  # Length of individual beam segments (mm)
        self.width = width  # Width of beam (mm) - along Y
        self.thickness = thickness  # Thickness of beam (mm) - along Z
        self.num_beams = num_beams  # Number of beam segments
        self.configuration = configuration  # 'parallel', 'series', 'serpentine'

# Calculate flexure properties using beam theory
def calculate_flexure_properties(params, material, force):
    """
    Calculate deflection and stiffness for a flexure design
    
    Returns:
        x_deflection: Deflection in X direction (mm)
        y_deflection: Deflection in Y direction (mm)
        z_deflection: Deflection in Z direction (mm)
        stiffness: Force/deflection (N/mm)
    """
    # Convert to meters for calculations
    l = params.length / 1000  # m
    w = params.width / 1000  # m
    t = params.thickness / 1000  # m
    
    # Second moment of area
    Iy = (w * t**3) / 12  # For bending around y-axis (affects x-z deflection)
    Iz = (t * w**3) / 12  # For bending around z-axis (affects x-y deflection)
    
    # Calculate stiffness for a cantilever beam
    k_x = 0  # No axial deflection in simple model
    
    # For parallel configuration, stiffness is multiplied by number of beams
    if params.configuration == 'parallel':
        k_y = (3 * material.young_modulus * Iz) / (l**3) * params.num_beams
        k_z = (3 * material.young_modulus * Iy) / (l**3) * params.num_beams
    
    # For serpentine configuration
    elif params.configuration == 'serpentine':
        # Improved model for serpentine configuration
        # In serpentine, the compliance is dominated by the segments in the proper orientation
        horizontal_segments = (params.num_beams + 1) // 2  # Ceiling division
        vertical_segments = params.num_beams // 2  # Floor division
        
        # For y-deflection (in XY plane), horizontal segments contribute
        if horizontal_segments > 0:
            k_y = (3 * material.young_modulus * Iz) / (l**3 * horizontal_segments)
        else:
            k_y = float('inf')  # Infinite stiffness if no segments
            
        # For z-deflection, vertical segments contribute
        if vertical_segments > 0:
            k_z = (3 * material.young_modulus * Iy) / (l**3 * vertical_segments)
        else:
            k_z = float('inf')  # Infinite stiffness if no segments
    
    # For series configuration
    else:  # series
        k_y = (3 * material.young_modulus * Iz) / (l**3 * params.num_beams)
        k_z = (3 * material.young_modulus * Iy) / (l**3 * params.num_beams)
    
    # Calculate deflections
    # For very high stiffness values, set deflection to 0
    if k_y == float('inf'):
        y_deflection = 0
    else:
        y_deflection = force / k_y * 1000  # Convert back to mm
        
    if k_z == float('inf'):
        z_deflection = 0
    else:
        z_deflection = force / k_z * 1000  # Convert back to mm
    
    # For x deflection, we use 0 in this simple model
    x_deflection = 0
    
    # Return the stiffness in the direction of interest (y-direction in XY plane)
    stiffness = k_y / 1000  # N/mm
    
    return x_deflection, y_deflection, z_deflection, stiffness

# Objective function for optimization
def objective_function(x, material, target_deflection, max_force, target_plane='xy'):
    """
    Objective function to minimize for flexure optimization
    x = [length, width, thickness, num_beams]
    
    Returns a penalty value - lower is better
    """
    length, width, thickness, num_beams = x
    num_beams = max(2, int(num_beams))  # Ensure at least 2 beams for serpentine
    
    # Create flexure parameters
    # For XY-plane motion, use serpentine
    configuration = 'serpentine'
    
    params = FlexureParameters(length, width, thickness, num_beams, configuration)
    
    # Calculate properties with the target force
    _, y_deflection, z_deflection, stiffness = calculate_flexure_properties(params, material, max_force)
    
    # Calculate total size based on configuration
    if configuration == 'serpentine':
        total_size_x = length * ((num_beams + 1) // 2)  # Ceiling division for horizontal segments
        total_size_y = width * (num_beams // 2)  # Floor division for vertical segments
        if num_beams == 1:  # Special case for single beam
            total_size_y = width
    else:
        # Default size calculation (fallback)
        total_size_x = length
        total_size_y = width * num_beams
    
    # Initialize penalty
    penalty = 0
    
    # Penalty for exceeding max size
    if total_size_x > 250 or total_size_y > 250:
        penalty += 10000 * (max(0, total_size_x - 250) + max(0, total_size_y - 250))
    
    # Penalty for not meeting minimum deflection target (make this a hard constraint)
    if y_deflection < target_deflection:
        penalty += 5000 * (target_deflection - y_deflection)
    
    # Severe penalty for z-deflection
    penalty += abs(z_deflection) * 500
    
    # Penalty for force being too high
    force_needed = stiffness * target_deflection
    if force_needed > max_force:
        penalty += 2000 * (force_needed - max_force)
    
    # For z-plane constraint, add specific penalty
    force_z = 3  # N
    _, _, z_defl_at_3N, _ = calculate_flexure_properties(params, material, force_z)
    penalty += 2000 * abs(z_defl_at_3N)  # Severe penalty for any z-deflection at 3N
    
    # Add a small penalty for unnecessarily complex designs (more beams)
    penalty += num_beams * 0.5
    
    # Minor penalty for very thin beams (manufacturing concern)
    if thickness < 0.5:
        penalty += 100 * (0.5 - thickness)
    
    # Encourage higher width-to-thickness ratio to reduce z-deflection
    # We want width >> thickness for beams that bend in XY plane
    width_thickness_ratio = width / thickness
    if width_thickness_ratio < 1.0:
        penalty += 1000 * (1.0 - width_thickness_ratio)
    
    return penalty

# Function to run the optimization
def optimize_flexure(material, target_deflection=38, max_force=2, max_size=250, target_plane='xy'):
    """
    Optimize flexure design based on parameters
    
    Returns the optimized parameters
    """
    # Initial guess: [length, width, thickness, num_beams]
    x0 = [80, 20, 1, 4]
    
    # Bounds for parameters
    # length: 10-200mm, width: 1-50mm, thickness: 0.1-5mm, num_beams: 2-10
    bounds = [(10, 200), (1, 50), (0.1, 5), (2, 10)]
    
    # Run the optimization with multiple starting points to avoid local minima
    best_result = None
    best_penalty = float('inf')
    
    starting_points = [
        [80, 20, 1, 4],
        [100, 30, 0.5, 6],
        [60, 40, 0.3, 8],
        [120, 15, 2, 4],
        [90, 25, 1.5, 6]
    ]
    
    for x0 in starting_points:
        result = minimize(
            objective_function,
            x0,
            args=(material, target_deflection, max_force, target_plane),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 200}
        )
        
        if result.fun < best_penalty:
            best_penalty = result.fun
            best_result = result
    
    # Extract the result
    length, width, thickness, num_beams = best_result.x
    num_beams = max(2, int(round(num_beams)))  # Ensure integer and at least 2
    
    # Create the optimized parameters with serpentine configuration for XY-plane motion
    optimized_params = FlexureParameters(length, width, thickness, num_beams, 'serpentine')
    
    return optimized_params, best_penalty

# Function to visualize the flexure design
def visualize_flexure(params, save_path=None):
    """Generate a simple visualization of the flexure design"""
    
    # Create a figure with proper size
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale factors for visualization
    length = params.length
    width = params.width
    thickness = params.thickness
    
    # Define colors
    beam_color = 'cornflowerblue'
    edge_color = 'navy'
    
    # Calculate the number of horizontal and vertical segments
    if params.configuration == 'serpentine':
        horizontal_segments = (params.num_beams + 1) // 2  # Ceiling division
        vertical_segments = params.num_beams // 2  # Floor division
    else:
        horizontal_segments = params.num_beams
        vertical_segments = 0
    
    # Draw the flexure based on configuration
    if params.configuration == 'serpentine':
        # Fixed base at the start
        # Draw a fixed base rectangle
        base_w = width * 1.5
        base_h = thickness * 3
        base_d = length * 0.2
        
        # Base vertices
        base_vertices = np.array([
            [0, -base_w/3, -base_h/2],
            [base_d, -base_w/3, -base_h/2],
            [base_d, base_w + width - base_w/3, -base_h/2],
            [0, base_w + width - base_w/3, -base_h/2],
            [0, -base_w/3, base_h/2],
            [base_d, -base_w/3, base_h/2],
            [base_d, base_w + width - base_w/3, base_h/2],
            [0, base_w + width - base_w/3, base_h/2],
        ])
        
        # Base faces defined as lists of vertex indices
        base_faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5],  # right
        ]

        # Loop over faces, convert vertex indices to coordinates, and create a polygon
        for face_indices in base_faces:
            # Get the 3D points [ (x1, y1, z1), (x2, y2, z2), ... ]
            face_coords = [base_vertices[idx] for idx in face_indices]
            poly = Poly3DCollection([face_coords], facecolors='gray', edgecolors='black', alpha=0.7)
            ax.add_collection3d(poly)
        
        # Starting position for the beam segments
        current_x = base_d
        current_y = width / 2  # Center of base
        
        # Draw each segment of the serpentine
        for i in range(params.num_beams):
            if i % 2 == 0:  # Horizontal segment
                # Horizontal beam (X-direction)
                start_x = current_x
                start_y = current_y - width/2
                end_x = current_x + length
                
                # Beam cuboid vertices
                vertices = np.array([
                    [start_x, start_y, -thickness/2],
                    [end_x, start_y, -thickness/2],
                    [end_x, start_y + width, -thickness/2],
                    [start_x, start_y + width, -thickness/2],
                    [start_x, start_y, thickness/2],
                    [end_x, start_y, thickness/2],
                    [end_x, start_y + width, thickness/2],
                    [start_x, start_y + width, thickness/2]
                ])
                
                # Update current position
                current_x = end_x
                # current_y remains the same for horizontal segments
                
            else:  # Vertical segment
                # Vertical beam (Y-direction)
                start_y = current_y - width/2
                end_y = current_y - width/2 + width
                
                # Beam cuboid vertices for vertical segment
                # Note the different dimensions for vertical segment
                vertices = np.array([
                    [current_x - thickness/2, start_y, -thickness/2],
                    [current_x + thickness/2, start_y, -thickness/2],
                    [current_x + thickness/2, end_y, -thickness/2],
                    [current_x - thickness/2, end_y, -thickness/2],
                    [current_x - thickness/2, start_y, thickness/2],
                    [current_x + thickness/2, start_y, thickness/2],
                    [current_x + thickness/2, end_y, thickness/2],
                    [current_x - thickness/2, end_y, thickness/2]
                ])
                
                # Update current position
                # current_x remains the same for vertical segments
                current_y = end_y + width
            
            # Beam faces for any segment
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # front
                [2, 3, 7, 6],  # back
                [0, 3, 7, 4],  # left
                [1, 2, 6, 5],  # right
            ]
            for face_indices in faces:
                face_coords = [vertices[idx] for idx in face_indices]
                poly = Poly3DCollection([face_coords], facecolors=beam_color, edgecolors=edge_color, alpha=0.8)
                ax.add_collection3d(poly)
    
    # Draw a stylized applied force arrow
    if params.configuration == 'serpentine':
        force_x = current_x
        force_y = current_y - width/2 + width/2
        
        # Draw arrow for force in X-Y plane
        arrow_length = length * 0.3
        ax.quiver(
            force_x, force_y, 0,  # Start position
            -arrow_length, 0, 0,  # Direction and length
            color='red', arrow_length_ratio=0.3, linewidth=3
        )
        
        # Add text label for force
        ax.text(force_x - arrow_length/2, force_y + width/2, 0, "Force (2N)", 
                color='red', fontsize=12, ha='center')
    
    # Set the labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(f'Flexure Design: {params.configuration.capitalize()} Configuration', fontsize=14)
    
    # Calculate total size for proper view
    if params.configuration == 'serpentine':
        total_x = base_d + length * horizontal_segments
        total_y = width * (vertical_segments + 1) + base_w
        total_z = max(thickness, base_h)
    else:
        total_x = length
        total_y = width * params.num_beams
        total_z = thickness
    
    # Set axis limits for better viewing
    ax.set_xlim(-total_x * 0.1, total_x * 1.1)
    ax.set_ylim(-total_y * 0.1, total_y * 1.1)
    ax.set_zlim(-total_z * 3, total_z * 3)  # Exaggerate Z for visibility
    
    # Set equal aspect ratio for all axes
    max_range = max([total_x, total_y, total_z * 6])
    mid_x = total_x / 2
    mid_y = total_y / 2
    mid_z = 0
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/6, mid_z + max_range/6)
    
    # Add annotations
    param_text = (
        f"Configuration: {params.configuration}\n"
        f"Beam length: {params.length:.2f} mm\n"
        f"Beam width: {params.width:.2f} mm\n"
        f"Beam thickness: {params.thickness:.2f} mm\n"
        f"Number of segments: {params.num_beams}"
    )
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to file if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

# Function to analyze and report on the flexure performance
def analyze_flexure(params, material):
    """Analyze the performance of a flexure design"""
    # Calculate properties for various forces
    forces = np.linspace(0, 5, 20)
    x_deflections = []
    y_deflections = []
    z_deflections = []
    
    for force in forces:
        x_defl, y_defl, z_defl, _ = calculate_flexure_properties(params, material, force)
        x_deflections.append(x_defl)
        y_deflections.append(y_defl)
        z_deflections.append(z_defl)
    
    # Calculate key metrics
    _, y_defl_at_2N, z_defl_at_2N, stiffness = calculate_flexure_properties(params, material, 2)
    _, y_defl_at_3N, z_defl_at_3N, _ = calculate_flexure_properties(params, material, 3)
    
    force_for_38mm = stiffness * 38
    
    # Calculate total size
    if params.configuration == 'serpentine':
        horizontal_segments = (params.num_beams + 1) // 2  # Ceiling division
        vertical_segments = params.num_beams // 2  # Floor division
        total_size_x = params.length * horizontal_segments
        total_size_y = params.width * (vertical_segments + (1 if vertical_segments == 0 else 0))
    elif params.configuration == 'parallel':
        total_size_x = params.length
        total_size_y = params.width * params.num_beams
    else:  # series
        total_size_x = params.length * params.num_beams
        total_size_y = params.width
    
    # Print analysis
    print(f"Flexure Design Analysis:")
    print(f"------------------------")
    print(f"Configuration: {params.configuration}")
    print(f"Number of beam segments: {params.num_beams}")
    print(f"Beam length: {params.length:.2f} mm")
    print(f"Beam width: {params.width:.2f} mm")
    print(f"Beam thickness: {params.thickness:.2f} mm")
    print(f"Total size: {total_size_x:.2f} mm x {total_size_y:.2f} mm")
    print(f"------------------------")
    print(f"Stiffness: {stiffness:.4f} N/mm")
    print(f"Force required for 38mm deflection: {force_for_38mm:.2f} N")
    print(f"Deflection at 2N: {y_defl_at_2N:.2f} mm in Y, {z_defl_at_2N:.4f} mm in Z")
    print(f"Deflection at 3N: {y_defl_at_3N:.2f} mm in Y, {z_defl_at_3N:.4f} mm in Z")
    print(f"------------------------")
    
    # Check if it meets requirements
    requirements_met = True
    
    if total_size_x > 250 or total_size_y > 250:
        print("❌ SIZE REQUIREMENT NOT MET: Exceeds 250mm x 250mm")
        requirements_met = False
    else:
        print("✅ Size requirement met: Within 250mm x 250mm")
    
    if y_defl_at_2N < 38:
        print(f"❌ DEFLECTION REQUIREMENT NOT MET: Deflection at 2N is {y_defl_at_2N:.2f}mm (less than 38mm)")
        requirements_met = False
    else:
        print(f"✅ Deflection requirement met: Deflection at 2N is {y_defl_at_2N:.2f}mm (≥38mm)")
    
    if abs(z_defl_at_3N) > 0.1:  # Stricter tolerance for z-deflection
        print(f"❌ Z-PLANE REQUIREMENT NOT MET: Z-deflection at 3N is {z_defl_at_3N:.4f}mm (should be ~0)")
        requirements_met = False
    else:
        print(f"✅ Z-plane requirement met: Z-deflection at 3N is {z_defl_at_3N:.4f}mm (approximately 0)")
    
    if requirements_met:
        print("\n🎉 ALL REQUIREMENTS MET!")
    else:
        print("\n⚠️ SOME REQUIREMENTS NOT MET")
    
    # Plot deflection vs force
    plt.figure(figsize=(10, 6))
    plt.plot(forces, y_deflections, 'b-', linewidth=2, label='Y-deflection (in XY plane)')
    plt.plot(forces, z_deflections, 'r--', linewidth=2, label='Z-deflection')
    plt.axhline(y=38, color='g', linestyle=':', label='Target deflection (38mm)')
    plt.axvline(x=2, color='k', linestyle=':', label='Min. force (2N)')
    plt.axvline(x=3, color='m', linestyle=':', label='Z-constraint force (3N)')
    
    plt.xlabel('Applied Force (N)')
    plt.ylabel('Deflection (mm)')
    plt.title('Flexure Deflection vs. Applied Force')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the graph to file
    plt.savefig('flexure_performance.png', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
    
    return requirements_met

# Main function to run the entire optimization and analysis
def main():
    # Create material
    aluminum = Material()
    
    print("Starting flexure optimization...")
    print("Target parameters:")
    print("- Maximum size: 250mm x 250mm")
    print("- Minimum deflection: 38mm")
    print("- Maximum force: 2N")
    print("- Deflection plane: XY only")
    print("- No Z deflection at 3N force")
    print("\nOptimizing design...")
    
    # Run the optimization
    optimized_params, penalty = optimize_flexure(
        aluminum,
        target_deflection=38,
        max_force=2,
        max_size=250,
        target_plane='xy'
    )
    
    print(f"\nOptimization complete (penalty value: {penalty:.2f})")
    
    # Save visualization to file
    visualize_flexure(optimized_params, 'flexure_design.png')
    print("\nFlexure visualization saved to 'flexure_design.png'")
    
    # Analyze the optimized design
    analyze_flexure(optimized_params, aluminum)
    print("\nPerformance graph saved to 'flexure_performance.png'")
    
    # Return the optimized parameters
    return optimized_params

# Run the optimization if this script is executed directly
if __name__ == "__main__":
    optimized_flexure = main()
