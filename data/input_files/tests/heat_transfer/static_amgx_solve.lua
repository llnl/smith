-- Comparison information
expected_temperature_l2norm = 2.02263
epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "ball",
    -- number of elements in the mesh
    approx_elements = 10000,
    -- serial and parallel refinement levels
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

-- Solver parameters
heat_transfer = {
    equation_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-6,
                abs_tol     = 1.0e-12,
                max_iter    = 200,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "AMGX",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-4,
            abs_tol     = 1.0e-8,
            max_iter    = 500,
            print_level = 1,
        },
    },

    -- polynomial interpolation order
    order = 2,

    -- material parameters
    materials = { { model = "LinearIsotropicConductor", kappa = 0.5, cp = 1.0, density = 1.0 }, },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            constant = 1.0
        },
    },
}
