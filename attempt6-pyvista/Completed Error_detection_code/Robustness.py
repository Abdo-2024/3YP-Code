import numpy as np
import matplotlib.pyplot as plt
"""
Inputting values from google spread sheet and just using the average value rounded up
"""
# Define the decimation values (x-axis labels)
decimation_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
n_groups = len(decimation_values)
x = np.arange(n_groups)
bar_width = 0.25  # Width of each bar

# ---------------------------
# Pyvista Data 
# Each row corresponds to a run, columns correspond to decimation values 0.3, 0.4, ..., 0.9.
pyvista_data = np.array([
    [80, 79, 79, 80, 78, 79, 77]
])
pyvista_avg = np.mean(pyvista_data, axis=0)
pyvista_std = np.std(pyvista_data, axis=0, ddof=0)

# ---------------------------
# Winding Number Data 
winding_data = np.array([
    [80, 80, 80, 80, 80, 80, 80]
])
winding_avg = np.mean(winding_data, axis=0)
winding_std = np.std(winding_data, axis=0, ddof=0)

# ---------------------------
# Alternate Winding Number Data 
alt_winding_data = np.array([
    [46, 45, 40, 38, 35, 34, 33]
])
alt_winding_avg = np.mean(alt_winding_data, axis=0)
alt_winding_std = np.std(alt_winding_data, axis=0, ddof=0)

# ---------------------------
# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Pyvista bars (shifted left)
rects1 = ax.bar(x - bar_width, pyvista_avg, bar_width,
                yerr=pyvista_std, capsize=5, label='Pyvista', color='#434173')
# Plot Winding Number bars (centered)
rects2 = ax.bar(x, winding_avg, bar_width,
                yerr=winding_std, capsize=5, label='Winding Number', color='#414F73')
# Plot the alternate Winding number bars (shifted right)
rects3 = ax.bar(x + bar_width, alt_winding_avg, bar_width,
                yerr=alt_winding_std, capsize=5, label='Winding number', color='#415F73')

# Label the axes and title
ax.set_xticks(x)
# Use the decimation values as x-axis tick labels
ax.set_xticklabels(decimation_values)
ax.set_xlabel('Decimation Value')
ax.set_ylabel('Detected Errors (out of 80)')
ax.set_title('Detected Errors vs. Decimation Value (80 Errors)')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save the figure as an SVG file
plt.savefig('detected_errors_vs_decimation.svg', format='svg')
plt.show()
