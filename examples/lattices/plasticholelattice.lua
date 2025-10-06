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


simulation_steps = 1000         -- Number of time steps in the simulation
simulation_time = 2.0          -- Total simulation time (seconds)
serial_refinement = 0          -- Number of mesh refinements in serial mode
parallel_refinement = 0        -- Number of mesh refinements in parallel mode
--[[ 
  Material Properties
  -------------------
  Physical properties for the solid material.
--]]

density = 1.0                  -- Material density (units depend on simulation)
lambda = 1.0                   -- First Lamé parameter (elasticity)
G = 1.0                        -- Shear modulus (second Lamé parameter)
mu = 1.0e-2

material_parameters = {
  key = {"density", "K", "G"}, -- List of parameter names ("K" is bulk modulus)
  values = {
    density,                   -- Material density
    (3.0 * lambda + 2.0 * G) / 3.0, -- Bulk modulus K, computed from Lamé parameters
    G,                        -- Shear modulus
    mu
  },
}

--[[ 
  Boundary Conditions
  -------------------
  Function defining time-dependent displacement applied at the top boundary.
--]]

function strain_function(t)
  local strain_rate = -10.0
  if t < 0 then
      return 0
  elseif t < 1.0 then
      return strain_rate * t
  elseif t < 2.0 then
      return strain_rate * (2.0 - t)
  else
      return 0
  end
end

function applied_displacement(x, y, z, t)
  return {0.0, strain_function(t), 0.0}
end
