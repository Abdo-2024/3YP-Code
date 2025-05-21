import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
"""
This script simulates a time-domain A-scan acquisition with a Gaussian optical pulse, adding surface and internal reflections plus noise, digitises the signal to a 12-bit ADC scale, computes the analytic signal via the Hilbert transform, estimates and subtracts the front-echo response to suppress the strong surface reflection (optionally reintroducing a weak layer echo), and plots the raw, ADC-captured, and echo-suppressed traces.
"""

# ----------------------- user‑editable parameters -----------------------
adc_bits      = 12                         # your detector resolution
fs            = 2.0e9                      # sample rate (2 GS/s ≈ 150 µm range per trace)
tspan         = 400e-9                     # 400 ns trace (depth ≈ 60 mm, exaggerated for visibility)
pulse_FWHM    = 7e-9                       # optical PSF FWHM 7 ns  ➜  ~0.5 mm coherence length
λ0            = 800e-9                     # central wavelength (only needed for phase calc)
R_front       = 0.04                       # |air‑resin| Fresnel amplitude  (4 %)
R_internal    = 1e-4                       # weak interface you care about
z_internal    = 0.5e-3                     # depth 3 mm inside
noise_rms     = 2e-5                       # relative shot noise (adjust)
# -----------------------------------------------------------------------

# derived constants
adc_fullscale = 2**adc_bits - 1
t = np.arange(0, tspan, 1/fs)              # time axis
c = 2.998e8                                 # speed of light

# analytic-system's "input pulse" (complex PSF); Gaussian envelope
sigma = pulse_FWHM / (2*np.sqrt(2*np.log(2)))
I  = np.exp(-(t-t[0])**2/(2*sigma**2)) * np.exp(1j*2*np.pi*(c/λ0)*(t-t[0]))

# true scene: surface + one internal reflector
scene = np.zeros_like(t, dtype=complex)
scene += R_front * I
delay_samples = int(2*z_internal/c * fs)    # two‑way delay in samples
scene[delay_samples:delay_samples+len(I)] += R_internal * I[:len(t)-delay_samples]

# add noise
scene += (noise_rms*np.random.randn(len(t))) + 1j*(noise_rms*np.random.randn(len(t)))

# convert to detector signal (magnitude) and clip to 12‑bit
raw = np.abs(scene)
raw /= raw.max()                 # normalise so surface ≈ 1
raw_adc = np.clip(raw * adc_fullscale, 0, adc_fullscale)

# ==== apply the paper’s surface‑echo suppression =======================
# 1. build analytic version of *measured* trace
xa = hilbert(raw_adc / adc_fullscale)       # back to 0–1 range, complex

# 2. estimate front‑echo peak index k_peak (simple argmax will do here)
k_peak = np.argmax(np.abs(xa))
# Mirror the first half to estimate full front‑echo shape
front_half = xa[:k_peak]
front_est  = np.concatenate((front_half, front_half[::-1]), axis=0)[:len(xa)]

# 3. subtract the modelled front echo
xa_clean = xa - front_est

# 4. (optional) put back a very weak resin‑layer echo (1/100 of internal)
thin_reflection = 1e-6 * front_est          # tweak as needed
xa_clean += thin_reflection

# ======================================================================

# plotting
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.title("Raw magnitude (before ADC)")
plt.plot(t*1e9, raw, 'k')
plt.ylabel("Amplitude")
plt.subplot(3,1,2)
plt.title("12‑bit captured trace (surface clipped)" )
plt.plot(t*1e9, raw_adc/adc_fullscale, 'r')
plt.ylabel("Amplitude")
plt.subplot(3,1,3)
plt.title("After synthetic front‑echo subtraction")
plt.plot(t*1e9, np.abs(xa_clean), 'g')
plt.xlabel("Time [ns]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
