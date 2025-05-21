// Define a module to import the main STL file
module main_model() {
    import("2x2_MN_Array_scaled.stl");
}

// Define lists of additive and subtractive noise spheres
additive_spheres = [
    // Format: [x, y, z, radius]
    [-0.3, 0.8, 0.5, 0.01],
    [-0.6, 0.2, 0.2, 0.05],
    [-0.1, 0.6, 0.1, 0.08]
];

subtractive_spheres = [
    [-0.5, 0.3, 0.5, 0.01],
    [-0.2, 0.5, 0.2, 0.05],
    [-0.5, 0.5, 0.4, 0.08]
];

// Module to add noise (simulate extra material)
module add_noise() {
    union() {
        for (s = additive_spheres) {
            translate([s[0], s[1], s[2]])
                sphere(r = s[3], $fn = 50);
        }
    }
}

// Module to subtract noise (simulate missing material)
module subtract_noise() {
    union() {
        for (s = subtractive_spheres) {
            translate([s[0], s[1], s[2]])
                sphere(r = s[3], $fn = 50);
        }
    }
}

// Combine the main model with additive noise and subtract the subtractive noise
difference() {
    union() {
        main_model();   // Call the module for the main model
        add_noise();
    }
    subtract_noise();
}
