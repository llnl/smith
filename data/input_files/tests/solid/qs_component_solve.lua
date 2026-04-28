-- Comparison information
expected_displacement_l2norm = 0.0054916
epsilon = 0.0001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/square_attribute.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 3,
    par_ref_levels = 0,
}

-- Solver parameters
solid = {
    equation_solver = {
        linear = {
            type = "direct",
            direct_options = {
                print_level = 0,
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-6,
            abs_tol     = 1.0e-8,
            max_iter    = 5000,
            print_level = 1,
        },
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters
    materials = { { model = "NeoHookean", mu = 0.25, K = 1.0, density = 1.0 }, },

    -- boundary condition parameters
    boundary_conds = {
        ['displacement_1'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {2},
            component = 0,
            scalar_function = function (v)
                return v.y * -1.0e-2
            end
        },
        ['displacement_2'] = {
            -- boundary attribute 2 (index 0) is fixed (Dirichlet) in all directions
            attrs = {1},
            vector_constant = {
                x = 0.0,
                y = 0.0,
                z = 0.0
            }
        }

    },
}
