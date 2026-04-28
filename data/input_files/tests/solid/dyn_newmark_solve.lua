-- Comparison information
expected_displacement_l2norm = 1.4225
expected_velocity_l2norm = 0.2252
epsilon = 0.0001

-- Simulation time parameters
dt      = 0.01
t_final = 0.01

main_mesh = {
    type = "box",
    elements = {x = 3, y = 1},
    size = {x = 8, y = 1},
    -- serial and parallel refinement levels
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

-- Solver parameters
solid = {
    equation_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-8,
                abs_tol     = 1.0e-12,
                max_iter    = 500,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "HypreAMG",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-8,
            abs_tol     = 1.0e-12,
            max_iter    = 500,
            print_level = 1,
        },
    },

    dynamics = {
        timestepper = "NewmarkBeta",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    materials = { { model = "NeoHookean", mu = 0., K = .0, density = 1.0 }, },

    -- initial conditions
    initial_displacement = {
        vector_constant = {
            x = 0.0,
            y = 0.0
        }
    },

    initial_velocity = {
        vector_constant = {
            x = 0.0,
            y = 1.0
        }
    },

}
