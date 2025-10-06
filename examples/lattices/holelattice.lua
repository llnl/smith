-- script.lua
-- print("Using Lua Script")

--[[ 
  Simulation Properties
  ---------------------
  General configuration and simulation control parameters.
--]]

problem_type = "nominal"          -- options [nominal, optimized, synthetic]
use_contact = false
-- output_location = "/usr/WS2/korner1/SentientMaterialSIProject/thermomechanics/build-dane-toss_4_x86_64_ib-clang@19.1.3-release/examples/test_output"


simulation_steps = 200         -- Number of time steps in the simulation
simulation_time = 1.0          -- Total simulation time (seconds)
serial_refinement = 0          -- Number of mesh refinements in serial mode
parallel_refinement = 1        -- Number of mesh refinements in parallel mode

--[[ 
  Material Properties
  -------------------
  Physical properties for the solid material.
--]]

density = 1.0                  -- Material density (units depend on simulation)
lambda = 1.0                   -- First Lamé parameter (elasticity)
G = 1.0                        -- Shear modulus (second Lamé parameter)

material_parameters = {
  key = {"density", "K", "G"}, -- List of parameter names ("K" is bulk modulus)
  values = {
    density,                   -- Material density
    (3.0 * lambda + 2.0 * G) / 3.0, -- Bulk modulus K, computed from Lamé parameters
    G,                        -- Shear modulus
  },
}

--[[ 
  Boundary Conditions
  -------------------
  Function defining time-dependent displacement applied at the top boundary.
--]]

function applied_displacement(x, y, z, t)
  local strain_rate = -10.0;  -- Strain rate for the boundary displacement
  -- local mag = 5.0             -- Magnitude of displacement
  -- Returns displacement vector applied to the top face at time t
  -- return {0.0, mag * math.sin(strain_rate * t), 0.0}
  return {0.0, strain_rate * t, 0.0}
end
