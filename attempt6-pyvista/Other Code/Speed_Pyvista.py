import numpy as np
import matplotlib.pyplot as plt

# Define the number of errors (x-axis) and PyVista timings from two runs (in seconds)
errors = np.array([10, 20, 40, 80, 100])
pyvista_run1 = np.array([0.4835, 0.6077, 0.7237, 1.0560, 2.2120])
pyvista_run2 = np.array([0.4761, 0.5626, 0.6668, 1.0268, 2.3409])

# Compute the average running time for each error count
mean_times = (pyvista_run1 + pyvista_run2) / 2

# Compute the sample standard deviation for each error count (n=2)
std_times = np.sqrt(((pyvista_run1 - mean_times)**2 + (pyvista_run2 - mean_times)**2) / (2 - 1))

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(errors, mean_times, yerr=std_times, marker='o', linestyle='-',
             color='#414f73ff', ecolor='k', label='PyVista')

# Label the axes, add a title and legend, and a grid for clarity
plt.xlabel('Number of Errors')
plt.ylabel('Time (seconds)')
plt.title('Close-up: PyVista Running Time vs. Number of Errors')
plt.legend()
plt.grid(True)

# Adjust the y-axis to include the last data point and its error bars
plt.ylim(0, 2.5)

# Save the figure as an SVG file and display the plot
plt.savefig('pyvista_closeup_errorbars_custom.svg', format='svg')
plt.show()
