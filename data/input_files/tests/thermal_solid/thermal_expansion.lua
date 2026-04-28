-- Simulation time parameters
t_final = 1.0
dt      = 1.0

expected_displacement_l2norm = 0.1301349
expected_velocity_l2norm = 0.0
expected_temperature_l2norm = 2.3424281
epsilon = 0.0001

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/onehex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 2,
    par_ref_levels = 0,
}

thermal_solid = {
    -- Solver parameters
    solid = {
        equation_solver = {
            linear = {
                type = "direct"
            },

            nonlinear = {
                rel_tol     = 1.0e-6,
                abs_tol     = 1.0e-8,
                max_iter    = 500,
                print_level = 1,
            },
        },

        -- polynomial interpolation order
        order = 1,

        -- neo-Hookean material parameters
        materials = { { model = "NeoHookean", mu = 0.25, K = 5.0, density = 1.0 }, },

        initial_displacement = {
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },

        initial_velocity = {
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },

        -- boundary condition parameters
        boundary_conds = {
            ['displacement_x'] = {
                attrs = {1},
                component = 0,
                constant = 0.0
            },
            ['displacement_y'] = {
                attrs = {2},
                component = 1,
                constant = 0.0,
            },
            ['displacement_z'] = {
                attrs = {3},
                component = 2,
                constant = 0.0,
            },
        },
    },

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
                    solver_type = "cg",
                    prec_type   = "JacobiSmoother",
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
        order = 1,

        -- material parameters
        materials = { { model = "LinearIsotropicConductor", kappa = 0.5, cp = 0.5, density = 0.5 }, },

        -- initial conditions
        initial_temperature = {
            constant = 2.0
        },

        -- boundary condition parameters
        boundary_conds = {
            ['temperature'] = {
                attrs = {1,2,3,4,5,6},
                constant = 2.0
            },
        },
    },

    coef_thermal_expansion = {
        constant = 0.1
    },

    reference_temperature = {
        constant = 1.0
    }

}
