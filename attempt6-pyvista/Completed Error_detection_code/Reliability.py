import numpy as np
import matplotlib.pyplot as plt
"""
Inputting values from google spread sheet and just using the average value rounded up
"""
# Define the error counts (X axis values) for the groups 
errors = np.array([0, 10, 20, 40, 80, 160, 320, 640])
n_groups = len(errors)
x = np.arange(n_groups)
bar_width = 0.25  # width of each bar

# ---------------------------
# Pyvista Data 
pyvista_data = np.array([
    [0, 9, 20, 40, 80, 155, 316, 638]
])
pyvista_avg = np.mean(pyvista_data, axis=0)
pyvista_std = np.std(pyvista_data, axis=0, ddof=0)

# ---------------------------
# Winding Number Data 
winding_data = np.array([
    [0, 10, 20, 40, 79, 160, 319, 640]
])
winding_avg = np.mean(winding_data, axis=0)
winding_std = np.std(winding_data, axis=0, ddof=0)

# ---------------------------
# Comparison Data 
comparison_data = np.array([
    [0, 5, 10, 20, 45, 80, 240, 350]
])
comparison_avg = np.mean(comparison_data, axis=0)
comparison_std = np.std(comparison_data, axis=0, ddof=0)

# ---------------------------
# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - bar_width, pyvista_avg, bar_width,
                yerr=pyvista_std, capsize=5, label='Pyvista', color='#434173')
rects2 = ax.bar(x, winding_avg, bar_width,
                yerr=winding_std, capsize=5, label='Winding Number', color='#414F73')
rects3 = ax.bar(x + bar_width, comparison_avg, bar_width,
                yerr=comparison_std, capsize=5, label='Comparison', color='#415F73')

ax.set_xticks(x)
ax.set_xticklabels(errors)
ax.set_xlabel('Number of Errors')
ax.set_ylabel('Averaged Detected Error')
ax.set_title('Averaged Detected Error vs. Number of Errors')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Set the y-axis ticks to the desired values and adjust the y limit to show error bars for the 640 bars
ax.set_yticks([0, 10, 20, 40, 80, 160, 320, 640])
ax.set_ylim(0, 700)

# Save the result as an SVG file
plt.savefig('averaged_detected_error_bar_chart_custom_y_ticks.svg', format='svg')
plt.show()
