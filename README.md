# 3YP-Code
This Contains all the code I used for my 3YP project, I apologies it is a little messy.
(Topology code might be missing I am not sure if I did it on windows)

---

# Mesh Error Analysis and Surface Topology Exploration

This repository contains code and data related to error analysis, surface topology exploration, and simulation of A-scan imaging from SD-OCT.

## Contents Overview

* **Error Analysis**:

  * Implementation of error detection algorithms for meshes.
  * Visualization of error models and detection results.
  * Creation of loops on mesh surfaces using SWN.

* **Surface Topology**:

  * Attempted topology modifications and algorithms.

* **SD-OCT Simulation**:

  * Scripts for simulating and analyzing A-scan imaging from SD-OCT data.
  * as well as simulating intensity plots
  * simulating a "3D Scan" of a microneedle  

* **Plotting and Visualization**:

  * Scripts for generating plots from averaged simulation runs (the runs are averaged on a google drive so sadly I can't include it here)
  * Visualization of error clusters and surface modifications. (this is usually done is most error code, I try to have it output a visual)

## Usage

Each folder aims to contain specific scripts and data related to its named functionality. Refer to each code file for detailed usage instructions.

## Folder Layout 
Numbering indects restarting from scratch as I am not sure what I broke :)

* **attempt1 to attempt6-pyvista**: Various attempts and iterations of error assignment and detection algorithms (+ other codes listed mostly in attempt6-pyvista)
* **Illustrations\_SWN**: Code for creating visual illustrations related to making loops on the surface of a mesh using the SWN algorithm.
* **SD-OCT**: Scripts for SD-OCT imaging simulation (specifically making a A-Scan from reflectors).
* **Matlab**: MATLAB script an attempt to simulate light propagation through the system.

The most complete of folder and the one containing the code that I used for my report is in **attempt6-pyvista**

small note all illustrations are attempts to get the SWN algorithm to work on my computer, and I was struggling getting ortools to work. 
Refernce for SWN at the end in Requirements  

Code/
├── attempt1/
│   ├── Microneedle_Array_Generator2.py
│   ├── Microneedle_Array_Generator3.py
│   ├── Spot (Incoherent Irradiance) Polarized.png
│   ├── Test2.ipynb
│   ├── Test.ipynb
│   └── myenv/
├── attempt2/
│   ├── 2x2_MN_Array_scaled.stl
│   ├── combined_deformed2.stl
│   ├── combined_deformed_dragon.stl
│   ├── combined_scaled_dragon.stl
│   ├── Deformation_Detection_and_Analysis.py
│   ├── Error_Assignment_4.py
│   ├── Error_assignments_attempts/
│   ├── Noise_RDS_2.py
│   ├── Noise_RDS_3.py
│   ├── Other STLs/
│   ├── Other tests/
│   ├── __pycache__/
│   ├── parameters.py
│   ├── poison_dragon_.stl
│   ├── Scaling.py
│   ├── sphere_data.csv
│   └── sphere_data_detection.csv
├── attempt3/
│   ├── 1250_polygon_sphere_100mm.STL
│   ├── 1.scad
│   ├── 2x2_MN_Array_scaled.stl
│   ├── 2x2_MN_Array_scaled+AddSubNoise.stl
│   ├── 2x2_MN_Array_scaled+Noise.stl
│   ├── 2x2_MN_Array_scaled_minusSpheres.stl
│   ├── Consistency_test.py
│   ├── deformed_spheres.stl
│   ├── Error_Assignment.py
│   ├── Error_Assignment2.py
│   ├── Error_Assignment_Auto.py
│   ├── Error_Assignment_Auto2.py
│   ├── Error_Assignment_Auto3.py
│   ├── Error_Assignment_Auto4.py
│   ├── Error_Assignment_Auto5.py
│   ├── Error_Assignment_Auto6.py
│   ├── Error_Assignment_Auto7.py
│   ├── Noise_RDS_2.py
│   ├── Noise_RDS_3.py
│   ├── Noise_RDS_3.S.py
│   ├── Scaling.py
│   ├── Subtraction.py
│   ├── Testing_subtratcion.2.py
│   ├── testing_subtraction.py
│   ├── '# Randomaly Generated errors.py'
│   ├── error_detection_barplot.png
│   ├── error_models/
│   ├── import trimesh.py
│   ├── import trimesh2.py
│   ├── import trimesh3.py
│   ├── import trimesh4.py
│   ├── import trimesh5.py
│   ├── import trimesh6.py
│   ├── import trimesh7.py
│   ├── modified_model_with_holes.stl
│   ├── output_stls/
│   ├── sphere_data.csv
│   ├── sphere_data_detection.csv
│   ├── swiss_cheese_model.stl
│   ├── swiss_cheese_model2.stl
│   ├── swiss_cheese_model3.stl
│   ├── swiss_cheese_model5.stl
│   ├── test.ipynb
│   ├── venv/
│   └── venv310/
├── attempt4/
│   ├── 2x2_MN_Array_scaled.stl
│   ├── import pymesh.py
│   ├── Noise_RDS_2.py
│   ├── Noise_V2.py
│   ├── output_stls/
│   ├── PyMesh/
│   ├── Scaling.py
│   └── venv/
├── attempt5/
│   └── PyMesh/
├── **attempt6-pyvista**/
│   ├── Accuracy code/
│   ├── Completed Error_detection_code/
│   ├── CSV/
│   ├── env/
│   ├── Figures/
│   ├── Flexure_Topology/
│   ├── mesh_output/
│   ├── oct_slices_output/
│   ├── Other Code/
│   ├── Samples/
│   └── voxel_output (output from code in folder other code for intensity scan)/
├── Illustrations_SWN/        # SWN code
│   └── SWN/
├── Illustrations_2_SWN/      # SWN code
│   ├── or-tools_amd64_fedora-39_cpp_v9.11.4210
│   ├── or-tools_amd64_fedora-39_cpp_v9.11.4210.tar.gz
│   └── SWN/
├── Illustrations_3_SWN/      # SWN code
│   └── SWN/
├── Illustrations_4_SWN/      # SWN code
│   ├── Running_Commmand.txt
│   └── SWN/
├── Matlab/
│   └── Propagation.m
├── SD-OCT/
│   ├── DeltaR.py
│   ├── R1vsR2.py
│   ├── SD-OCT_SIM_Book.py
│   ├── SD-OCT_SIM.py
│   ├── T1.py
│   ├── test.py
│   └── venv/
├── SD-OCT_SIM/               # (merged into SD-OCT above)
└── Part Studio 1.stl


## Requirements

* Python 3.x
* For SWN refer to:
@article{Feng:2023:WND,
    author = {Feng, Nicole and Gillespie, Mark and Crane, Keenan},
    title = {Winding Numbers on Discrete Surfaces},
    year = {2023},
    issue_date = {August 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {42},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3592401},
    doi = {10.1145/3592401},
    journal = {ACM Trans. Graph.},
    month = {jul},
    articleno = {36}

## Owner
* \[AbdoAllah Mohammad]

## License
MIT
