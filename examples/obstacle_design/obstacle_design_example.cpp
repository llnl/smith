// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file obstacle_design_example.cpp
 *
 * @brief Obstacle Design example
 *
 * Intended to show how to solve a problem with the HomotopySolver.
 * The example problem solved is an inertia relief problem.
 */

#include <format>

#include "smith/smith.hpp"

#include "mfem.hpp"

// ContinuationSolver headers
#include "continuationsolvers/problems/ObstacleProblems.hpp"
#include "continuationsolvers/problems/OptProblems.hpp"
#include "continuationsolvers/problems/MPECProblems.hpp"
#include "continuationsolvers/solvers/MPECSolver.hpp"

#include "axom/sidre.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;
static constexpr int dim = 3;
static constexpr int order = 1;

using StateSpace = smith::H1<order>;
//using SolidWeakFormT = smith::SolidWeakForm<disp_order, dim, smith::Parameters<DensitySpace>>;


double fRhs(const mfem::Vector &x);
double flat_obstacle(const mfem::Vector &x);

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Command line arguments
  // Mesh options
  double xlength = 0.5;
  double ylength = 0.7;
  double zlength = 0.3;
  int nx = 6;
  int ny = 4;
  int nz = 4;

  // Solver options
  double nonlinear_absolute_tol = 1e-6;
  int nonlinear_max_iterations = 50;
  // Handle command line arguments
  axom::CLI::App app{"Inertial relief."};
  // Mesh options
  app.add_option("--xlength", xlength, "extent along x-axis")
      ->default_val("0.5")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--ylength", ylength, "extent along y-axis")
      ->default_val("0.7")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--zlength", zlength, "extent along z-axis")
      ->default_val("0.3")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  int nprocs;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_dynamics");

  std::shared_ptr<smith::Mesh> mesh;
  //std::vector<smith::FiniteElementState> states;
  //std::vector<smith::FiniteElementState> params;
  //std::vector<std::shared_ptr<smith::ScalarObjective>> constraints;

  mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(nx, ny, nz, element_shape, xlength, ylength, zlength), "this_mesh_name", 0, 0);

  auto [Vh, _] = smith::generateParFiniteElementSpace<StateSpace>(&mesh->mfemParMesh());
  mfem::Array<int> ess_tdof_list;
  //mfem::Array<int> ess_bdr;
  //if (mesh->mfemParMesh().bdr_attributes.Size())
  //{
  //   ess_bdr.SetSize(mesh->mfemParMesh().bdr_attributes.Max());
  //   ess_bdr[0] = 1;
  //   Vh->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  //}
  int dimU = Vh->GetTrueVSize();
  mfem::Vector uDC(dimU); uDC = 0.0;
  ParamObstacleProblem problem(Vh.get(), &fRhs, &flat_obstacle, ess_tdof_list, uDC);
  
  ObstacleDesignProblem designproblem(&problem);
  int dimPrimal = designproblem.GetDimU();
  mfem::Vector X0(dimPrimal); X0 = 0.0;
  mfem::Vector Xf(dimPrimal); Xf = 0.0;
  MPECSolver designoptimizer(&designproblem);
  designoptimizer.SetTol(1.e-5);
  designoptimizer.SetBarrierParameter(1.e-3);
  designoptimizer.SetMaxIter(30);
  designoptimizer.CheckLinearSystemResiduals();
  designoptimizer.RegularizePrimalHessian(1.e-10);
  designoptimizer.Mult(X0, Xf);
  



  //smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  //smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  //smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  //smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());
  //std::unique_ptr<smith::FiniteElementState> shape_disp =
  //    std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());

  //velo = 0.0;
  //accel = 0.0;

  //states = {disp, velo, accel};
  //params = {density};

  //std::string physics_name = "solid";

  //// construct residual
  //auto solid_mechanics_weak_form =
  //    std::make_shared<SolidWeakFormT>(physics_name, mesh, states[FIELD::DISP].space(), getSpaces(params));

  //SolidMaterial mat;
  //mat.K = 1.0;
  //mat.G = 0.5;
  //solid_mechanics_weak_form->setMaterial(smith::DependsOn<0>{}, mesh->entireBodyName(), mat);

  //// apply some traction boundary conditions
  //std::string surface_name = "side";
  //mesh->addDomainOfBoundaryElements(surface_name, smith::by_attr<dim>(1));
  //solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

  //smith::tensor<double, dim> constant_force{};
  //for (int i = 0; i < dim; i++) {
  //  constant_force[i] = 1.e0;
  //}

  //solid_mechanics_weak_form->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
  //  return smith::tuple{constant_force, 0.0 * smith::get<smith::DERIVATIVE>(x)};
  //});

  //// construct constraints
  //params[0] = 1.;

  //using ObjectiveT =
  //    smith::FunctionalObjective<dim, smith::Parameters<VectorSpace, DensitySpace>>;  // functional objective on
  //                                                                                    // displacement/density

  //double time = 0.0;
  //double dt = 1.0;
  //smith::TimeInfo time_info(time, dt, 0);
  //auto all_states = getConstFieldPointers(states, params);
  //auto objective_states = {all_states[FIELD::DISP], all_states[FIELD::DENSITY]};

  //ObjectiveT::SpacesT param_space_ptrs{&all_states[FIELD::DISP]->space(), &all_states[FIELD::DENSITY]->space()};

  //ObjectiveT mass_objective("mass constraining", mesh, param_space_ptrs);

  //mass_objective.addBodyIntegral(smith::DependsOn<1>{}, mesh->entireBodyName(),
  //                               [](double /*t*/, auto /*X*/, auto RHO) { return get<smith::VALUE>(RHO); });
  //double mass = mass_objective.evaluate(time_info, shape_disp.get(), objective_states);

  //smith::tensor<double, dim> initial_cg;  // center of gravity

  //for (int i = 0; i < dim; ++i) {
  //  auto cg_objective = std::make_shared<ObjectiveT>("translation " + std::to_string(i), mesh, param_space_ptrs);
  //  cg_objective->addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
  //                                [i](double
  //                                    /*time*/,
  //                                    auto X, auto U, auto RHO) {
  //                                  return (get<smith::VALUE>(X)[i] + get<smith::VALUE>(U)[i]) * get<smith::VALUE>(RHO);
  //                                });
  //  initial_cg[i] = cg_objective->evaluate(time_info, shape_disp.get(), objective_states) / mass;

  //  constraints.push_back(cg_objective);
  //}

  //for (int i = 0; i < dim; ++i) {
  //  auto center_rotation_objective =
  //      std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh, param_space_ptrs);
  //  center_rotation_objective->addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
  //                                             [i, initial_cg](double /*time*/, auto X, auto U, auto RHO) {
  //                                               auto u = get<smith::VALUE>(U);
  //                                               auto x = get<smith::VALUE>(X) + u;
  //                                               auto dx = x - initial_cg;
  //                                               auto x_cross_u = smith::cross(dx, u);
  //                                               return x_cross_u[i] * get<smith::VALUE>(RHO);
  //                                             });
  //  constraints.push_back(center_rotation_objective);
  //}

  //// initialize displacement
  //states[FIELD::DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
  //  auto u = 0.0 * x;
  //  return u;
  //});

  //auto writer = createParaviewWriter(mesh->mfemParMesh(), objective_states, "inertia_relief");
  //if (visualize) {
  //  writer.write(0, 0.0, objective_states);
  //}
  //auto non_const_states = getFieldPointers(states, params);
  //// create an inertial relief problem
  //InertialReliefProblem problem({non_const_states[FIELD::DISP], non_const_states[FIELD::DENSITY]}, non_const_states,
  //                              mesh, solid_mechanics_weak_form, constraints);

  //// optimization variables
  //auto X0 = problem.GetOptimizationVariable();
  //auto Xf = problem.GetOptimizationVariable();

  //// define a homotopy solver for the inertia relief problem
  //HomotopySolver solver(&problem);
  //// set solver options
  //solver.SetTol(nonlinear_absolute_tol);
  //solver.SetMaxIter(nonlinear_max_iterations);
  //solver.EnableRegularizedNewtonMode();
  //// solve the inertia relief problem
  //solver.SetPrintLevel(2);
  //solver.Mult(X0, Xf);
  //// extract displacement and Lagrange multipliers
  //mfem::Vector displacement_sol = problem.GetDisplacement(Xf);
  //mfem::Vector multiplier_sol = problem.GetLagrangeMultiplier(Xf);
  //bool converged = solver.GetConverged();
  //SLIC_ERROR_ROOT_IF(!converged, "Homotopy solver did not converge");
  //double displacement_norm = mfem::GlobalLpNorm(2, displacement_sol.Norml2(), MPI_COMM_WORLD);
  //double multiplier_norm = mfem::GlobalLpNorm(2, multiplier_sol.Norml2(), MPI_COMM_WORLD);
  //SLIC_INFO_ROOT(std::format("||displacement|| = {}", displacement_norm));
  //SLIC_INFO_ROOT(std::format("||multiplier|| = {}", multiplier_norm));

}

double fRhs(const mfem::Vector &x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (std::pow(x(0),3)- 1.5*std::pow(x(0),2.) - 6 * x(0) + 3.) + (1. + std::pow(2.*M_PI,2))*std::cos(2.*M_PI*x(0));
  return fx;
}

double flat_obstacle(const mfem::Vector &/*x*/)
{
  return 0.0;
}
