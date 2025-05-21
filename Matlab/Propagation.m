%% Beam Propagation Simulation Using Angular Spectrum Method
% Using the provided equations:
%   Lateral resolution (FWHM): d_x = 2 ln2 * w0
%   Thus, w0 = d_x/(2 ln2)
%   Parameters:
%     Wavelength lambda = 787.3 nm
%     NA = 0.0203
%     Measured FWHM, d_x = 17.11 µm

% Clear workspace and close figures
clear; close all; clc;

%% System Parameters
lambda = 787.3e-9;       % wavelength in meters
d_x = 17.11e-6;          % lateral resolution (FWHM) in meters
NA = 0.0203;             % numerical aperture

% Compute beam waist (1/e^2 radius) using provided relation:
ln2 = log(2);
w0 = d_x / (2*ln2);      % beam waist in meters
fprintf('Calculated beam waist w0 = %.2e m\n', w0);

% Check consistency with diffraction-limit formula:
w0_diff = lambda / (pi * NA);
fprintf('Diffraction-limited beam waist w0_diff = %.2e m\n', w0_diff);

% Compute Rayleigh range (for reference)
z_R = pi * w0^2 / lambda;  % Rayleigh range in meters
fprintf('Rayleigh range z_R = %.2e m\n', z_R);
% The confocal parameter (DOF) is approx 2*z_R.

%% Spatial Grid Setup
% Define simulation window that covers several beam widths.
gridSize = 200e-6;       % grid size (m) - adjust as needed
N = 512;                 % number of grid points per dimension
dx_grid = gridSize / N;  % spatial sampling interval (m)
x = linspace(-gridSize/2, gridSize/2 - dx_grid, N);
y = x;                   % square grid
[X, Y] = meshgrid(x, y);

%% Initial Field: Gaussian Beam at Waist (z=0)
E0 = exp( - (X.^2 + Y.^2) / (w0^2) );
% (The amplitude factor is set to 1)

%% Prepare FFT Spatial Frequency Grids
dk = 2*pi / gridSize;
kx = (-N/2:N/2-1) * dk;  % frequency coordinates (rad/m)
ky = kx;
[KX, KY] = meshgrid(kx, ky);
k0 = 2*pi/lambda;  % wave number

%% Angular Spectrum Propagation Function
propagate = @(E_in, z) ifftshift(ifft2( fft2(fftshift(E_in)) .* ...
    exp(1i * sqrt(max(0, k0^2 - KX.^2 - KY.^2)) * z ));

%% Propagation Simulation Over z
% Define z range: from -0.5 mm to +0.5 mm (relative to focus)
z_values = linspace(-0.5e-3, 0.5e-3, 21); % 21 steps
FWHM_vs_z = zeros(size(z_values));  % to store measured FWHM at each z

% We analyze the intensity profile along x at y=0.
center_idx = round(N/2);

figure;
hold on;
colors = jet(length(z_values));
for idx = 1:length(z_values)
    z = z_values(idx);
    % Propagate field to distance z
    E_z = propagate(E0, z);
    I_z = abs(E_z).^2;
    
    % Extract 1D intensity profile along x at y=0
    I_profile = I_z(center_idx, :);
    I_profile = I_profile / max(I_profile);  % normalize
    
    % Estimate FWHM: find indices where intensity is >= 0.5
    indices = find(I_profile >= 0.5);
    if ~isempty(indices)
        FWHM = (indices(end) - indices(1)) * dx_grid;  % in meters
    else
        FWHM = NaN;
    end
    FWHM_vs_z(idx) = FWHM * 1e6; % store in µm
    
    % Plot profiles for selected z values (e.g., z=-0.5mm, 0, 0.5mm)
    if idx == 1 || idx == ceil(length(z_values)/2) || idx == length(z_values)
        plot(x*1e6, I_profile, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('z = %.0f µm', z*1e6));
    end
end
hold off;
xlabel('x (µm)');
ylabel('Normalized Intensity');
title('Intensity Profiles along x at Selected z Positions');
legend;
grid on;

%% Plot FWHM vs. Propagation Distance z
figure;
plot(z_values*1e6, FWHM_vs_z, 'o-', 'LineWidth', 1.5);
xlabel('Propagation Distance z (µm)');
ylabel('Estimated FWHM (µm)');
title('Beam Spot Size (FWHM) vs. Propagation Distance');
grid on;

%% (Optional) Simulate Passage Through a 0.5 mm Glass Slide
% Here we simulate propagation through a medium of thickness 0.5 mm with refractive index n = 1.52.
n_glass = 1.52;
thickness_glass = 0.5e-3;   % in meters
% In the glass, the optical path length is increased by a factor n.
z_glass = thickness_glass * n_glass;
E_afterGlass = propagate(E0, z_glass);
I_afterGlass = abs(E_afterGlass).^2;

figure;
imagesc(x*1e6, y*1e6, I_afterGlass);
axis image; colormap('inferno');
xlabel('x (µm)');
ylabel('y (µm)');
title(sprintf('Intensity After Propagation Through %.1f mm Glass (n=%.2f)', thickness_glass*1e3, n_glass));
colorbar;

