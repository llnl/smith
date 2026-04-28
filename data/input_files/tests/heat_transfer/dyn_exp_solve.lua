-- Comparison information
expected_temperature_l2norm = 2.6493029
epsilon = 0.00001

-- Simulation time parameters
dt      = 0.0001
t_final = 0.001

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/star.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 1,
}

temp_func = function (v)
    if v:norm() < 0.5 then return 2.0 else return 1.0 end
end

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

    dynamics = {
        timestepper = "ForwardEuler",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 2,

    -- material parameters
    materials = { { model = "LinearIsotropicConductor", kappa = 0.5, cp = 0.5, density = 0.5 }, },

    -- initial conditions
    initial_temperature = {
        scalar_function = temp_func
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            scalar_function = temp_func
        },
    },
}
