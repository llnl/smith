-- script.lua
-- print("Using Lua Script")

--[[ 
  Simulation Properties
  ---------------------
  General configuration and simulation control parameters.
--]]

problem_type = "optimized"          -- options [nominal, optimized, synthetic]
-- output_location = "/usr/WS2/korner1/SentientMaterialSIProject/thermomechanics/build-dane-toss_4_x86_64_ib-clang@19.1.3-release/examples/test_output"


simulation_steps = 200         -- Number of time steps in the simulation
simulation_time = .0000000020          -- Total simulation time (seconds)
serial_refinement = 0          -- Number of mesh refinements in serial mode
parallel_refinement = 0        -- Number of mesh refinements in parallel mode

--[[ 
  Material Properties
  -------------------
  Physical properties for the thermomechanical solid material.
--]]

rho        = 7850             -- Density, kg/m³
E0         = 1                -- Young's modulus, Pa
E          = 210e9            -- Young's modulus, Pa
nu         = 0.3              -- Poisson's ratio, dimensionless
c          = 500              -- Specific heat, J/(kg·K)
alpha0     = 1                -- Thermal expansion, 1/K
alpha      = 1.2e-5           -- Thermal expansion, 1/K
theta_ref  = 0                -- Reference temperature, K (20°C)
k0         = 1e-03            -- Thermal conductivity, W/(m·K)
k          = 45               -- Thermal conductivity, W/(m·K)

material_parameters = {
  key = {"rho", "E0", "E", "nu", "c", "alpha0", "alpha", "theta_ref", "k0", "k"}, -- List of parameter names 
  values = {
    rho,
    E0,
    E,
    nu,
    c,
    alpha0,
    alpha,
    theta_ref,
    k0,
    k
  },
}

--[[ 
  Boundary Conditions
  -------------------
  Function defining time-dependent displacement applied at the top boundary.
--]]

function applied_displacement(x, y, z, t)
  local strain_rate = -0.0;  -- Strain rate for the boundary displacement
  local mag = 5.0             -- Magnitude of displacement
  -- Returns displacement vector applied to the top face at time t
  return {0.0, mag * math.sin(strain_rate * t), 0.0}
end