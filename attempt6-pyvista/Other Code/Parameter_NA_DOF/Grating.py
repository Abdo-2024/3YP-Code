import numpy as np

# Define constants
lambda0 = 787.3e-9  # central wavelength in meters
bandwidth = 26e-9  # assume 100 nm bandwidth, adjust as needed
m = 1  # diffraction order

# Candidate grating parameters (in lines per mm)
groove_densities = [600, 800, 1200]  # example candidates
f_imaging = 50e-3  # focal length of imaging lens in spectrometer, adjust as needed

for density in groove_densities:
    d = 1/(density*1e3)  # grating period in meters (since density is per mm)
    # For Littrow, assume: 2d sin(theta_B) = m*lambda0
    theta_B = np.arcsin(m*lambda0/(2*d))
    # Calculate angular dispersion:
    dtheta_dlambda = m/(2*d*np.cos(theta_B))
    # Linear dispersion on the detector:
    linear_dispersion = f_imaging * dtheta_dlambda  # in meters per meter (i.e. dimensionless, convert to nm/nm)
    linear_dispersion_nm = linear_dispersion * 1e9  # nm per nm, effectively unitless, but tells you the scale
    print(f"Grating {density} lines/mm: Blaze angle = {np.degrees(theta_B):.2f} deg, linear dispersion ~ {linear_dispersion_nm:.2f} nm per nm")
    
    # You would then compute the total spectral spread on the detector:
    # Detector width = 14.34 mm. So, if the linear dispersion is L (m/nm), then the spread is:
    spread = 14.34e-3 / f_imaging / dtheta_dlambda  # in nm, roughly
    print(f"Total spectral spread on detector: {spread:.2f} nm")
