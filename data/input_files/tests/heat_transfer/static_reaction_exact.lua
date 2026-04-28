-- Comparison information
exact_function = function (v)
  return v.x^2 * v.y
end

minus_laplacian_temp = function (v)
  return -2.0 * v.y
end

reaction = function (temp)
  return temp^2 + 5.0
end

d_reaction = function (temp)
  return 2.0 * temp
end

scale_function = function (v)
    return math.sin(v.x) + math.cos(v.y)
end

exact_solution = {
  scalar_function = exact_function
}

epsilon = 0.00001

-- Simulation time parameters
dt      = 1.0

main_mesh = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/star.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 1,
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
                solver_type = "cg",
                prec_type   = "JacobiSmoother",
            },
        },

        nonlinear = {
            rel_tol     = 1.0e-8,
            abs_tol     = 1.0e-12,
            max_iter    = 500,
            print_level = 1,
        },
    },

    -- polynomial interpolation order
    order = 3,

    -- material parameters
    materials = { { model = "LinearIsotropicConductor", kappa = 1.0, cp = 1.0, density = 1.0 }, },

    -- add a nonlinear reaction
    nonlinear_reaction = {
        reaction_function = reaction,
        d_reaction_function = d_reaction,
        scale = {
            scalar_function = scale_function
        }
    },

    source = {
        scalar_function = function (v)
            return scale_function(v) * reaction(exact_function(v)) + minus_laplacian_temp(v)
        end
    },

    -- boundary condition parameters
    boundary_conds = {
        ['temperature'] = {
            attrs = {1},
            scalar_function = exact_function
        },
    },
}
