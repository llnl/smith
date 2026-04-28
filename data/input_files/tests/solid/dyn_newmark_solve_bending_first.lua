-- Comparison information
expected_displacement_l2norm = 1.4225
expected_velocity_l2norm = 0.2252
epsilon = 0.0001


main_mesh = {
    type = "box",
    elements = {x = 3, y = 1},
    size = {x = 8., y = 1.},
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

-- material property helper functions
function linear_mu(E, nu)
    return 0.5 * E / (1. + nu);
end

function linear_bulk(E, nu)
    return E / ( 3. * (1. - 2.*nu));
end

-- Problem parameters
E = 17.e6;
nu = 0.3;
r = 1.; -- density
g = -32.3; -- gravity
i = 1./12 * main_mesh.size.y * main_mesh.size.y * main_mesh.size.y;
m = main_mesh.size.y * r;
omega = (1.875) * (1.875) * math.sqrt(E * i / (m * main_mesh.size.x * main_mesh.size.x * main_mesh.size.x * main_mesh.size.x));
period = 2. * math.pi / omega;

-- Simulation time parameters
t_final = 2. * period;
nsteps = 40;
dt      = t_final / nsteps;

print("t_final = " .. t_final);

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
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-7,
            abs_tol     = 1.0e-10,
            max_iter    = 500,
            print_level = 1,
        },
    },

    dynamics = {
        timestepper = "BackwardEuler",
        enforcement_method = "RateControl",
    },

    -- polynomial interpolation order
    order = 1,

    -- neo-Hookean material parameters corresponding to linear elasticity
    materials = { { model = "NeoHookean", mu = linear_bulk(E, nu)/2., K = linear_bulk(E, nu)/2., density = 1.0 }, },

    -- initial conditions
    initial_displacement = {
        vector_constant = {
            x = 0.0,
            y = 0.0
        }
    },   

    -- boundary condition parameters
    boundary_conds = {
        ['displacement'] = {
            -- boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
            attrs = {2},
            vector_constant = {
                x = 0.0,
                y = 0.0
            }
        },
    },



}
