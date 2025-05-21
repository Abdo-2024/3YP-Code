import matplotlib.pyplot as plt
import numpy as np
"""
Inputting values from google spread sheet and just using the average value rounded up
"""
# Define the error counts (x-axis values)
errors = np.array([10, 20, 40, 80, 100])

# Pyvista data (two runs)
pyvista_run1 = np.array([0.4835, 0.6077, 0.7237, 1.0560, 2.2120])
pyvista_run2 = np.array([0.4761, 0.5626, 0.6668, 1.0268, 2.3409])
pyvista_data = np.vstack((pyvista_run1, pyvista_run2))
pyvista_avg = np.mean(pyvista_data, axis=0)
pyvista_std = np.std(pyvista_data, axis=0, ddof=0)

# Winding Number data (two runs)
winding_run1 = np.array([4.4742, 6.6951, 11.9119, 27.6714, 42.8662])
winding_run2 = np.array([4.2786, 7.3959, 12.6574, 21.2080, 29.3436])
winding_data = np.vstack((winding_run1, winding_run2))
winding_avg = np.mean(winding_data, axis=0)
winding_std = np.std(winding_data, axis=0, ddof=0)

# Comparison data (two runs)
comparison_run1 = np.array([28.2086, 28.9537, 30.8368, 27.8471, 30.2165])
comparison_run2 = np.array([27.891, 28.8645, 26.7488, 27.1216, 31.2115])
comparison_data = np.vstack((comparison_run1, comparison_run2))
comparison_avg = np.mean(comparison_data, axis=0)
comparison_std = np.std(comparison_data, axis=0, ddof=0)

# Set up the figure
plt.figure(figsize=(8, 6))

# Plot Pyvista with error bars:
plt.errorbar(errors, pyvista_avg, yerr=pyvista_std, fmt='o', color='#414f73ff', 
             linestyle='--', capsize=5, ecolor='black', label='Pyvista')

# Plot Winding Number with error bars:
plt.errorbar(errors, winding_avg, yerr=winding_std, fmt='o', color='#414f73ff', 
             linestyle='-.', capsize=5, ecolor='black', label='Winding Number')

# Plot Comparison with error bars:
plt.errorbar(errors, comparison_avg, yerr=comparison_std, fmt='o', color='#414f73ff', 
             linestyle='-', capsize=5, ecolor='black', label='Comparison')

# Labeling the plot
plt.xlabel('Number of Errors')
plt.ylabel('Time (seconds)')
plt.title('Running Time vs Number of Errors')
plt.legend()
plt.grid(True)

# Save the figure as an SVG
plt.savefig('runtime_vs_errors_with_errorbars.svg', format='svg')
plt.show()
