import pandas as pd
import matplotlib.pyplot as plt
"""
This script uses pandas to load time‐series data (attempting to read an .xlsx file via read_csv), renames an automatically generated column to “Time”, and plots “Voice Coil Force” against “Time” with markers and grid. It then saves the resulting figure as an SVG file and displays it using Matplotlib.
"""

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('/home/a/Downloads/CAD/Stage/VC380_Data_2.xlsx')

# Rename the "Unnamed: 3" column to "Time" for clarity
df = df.rename(columns={'Unnamed: 3': 'Time'})

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(df['Time'], df['Voice Coil Force'], marker='o', linestyle='-', color='b')
plt.xlabel('Time')
plt.ylabel('Voice Coil Force')
plt.title('Voice Coil Force vs Time')
plt.grid(True)

# Save the plot as an SVG file
plt.savefig('plot.svg', format='svg')

# Display the plot
plt.show()
