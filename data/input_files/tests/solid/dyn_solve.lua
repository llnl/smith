-- Comparison information
expected_displacement_l2norm = 1.4225
expected_velocity_l2norm = 0.2252
epsilon = 0.0001

-- Simulation time parameters
dt      = 1.0
t_final = 6.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

-- Solver parameters
solid = {
    equation_solver = {
        linear = {
            type = "iterative",
            iterative_options = {
                rel_tol     = 1.0e-4,
                abs_tol     = 1.0e-8,
                max_iter    = 500,
                print_level = 0,
                solver_type = "gmres",
                prec_type   = "HypreAMG",
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
        timestepper = "AverageAcceleration",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    materials = { { model = "NeoHookean", mu = 0.25, K = 5.0, density = 1.0 }, },

    -- initial conditions
    initial_displacement = {
        vector_constant = {
            x = 0.0,
            y = 0.0,
            z = 0.0
        }
    },

    initial_velocity = {
        vector_function = function (v)
            x = v.x
            s = 0.1 / 64
            first = -s * x * x
            last = s * x * x * (8.0 - x)
            if v.dim == 2 then
                return Vector.new(first, last)
            else
                return Vector.new(first, 0, last)
            end
        end 
    },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {1},
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        },
    },
}
