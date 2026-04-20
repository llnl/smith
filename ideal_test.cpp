// 2. BackpropagateThroughPhysics
TEST_F(ThermoMechanicsMeshFixture, BackpropagateThroughPhysics)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto solid_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");
  FieldType<L2<0>> youngs_modulus("youngs_modulus");

  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  QuasiStaticSecondOrderTimeIntegrationRule disp_rule;
  BackwardEulerFirstOrderTimeIntegrationRule temp_rule;
  auto param_fields = registerParameterFields(youngs_modulus);
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order>(field_store, disp_rule);
  auto thermal_fields = registerThermalFields<dim, temperature_order>(field_store, temp_rule);

  auto [solid_system, solid_cycle_zero_system, solid_end_step_systems] =
      buildSolidMechanicsSystem<dim, displacement_order, disp_rule>(
          std::make_shared<SystemSolver>(solid_block_solver), solid_opts, param_fields, solid_fields, thermal_fields);

  auto [thermal_system, thermal_cycle_zero_system, thermal_end_step_systems] =
      buildThermalSystem<dim, temperature_order, temp_rule, disp_rule>(
          std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts, param_fields, thermal_fields,
          solid_fields);

  auto coupled = combineSystems(solid_system, thermal_system);
  auto coupled_cycle_zero_system = combineSystems(solid_cycle_zero_system, thermal_cycle_zero_system);
  auto end_step_systems = combineSystems(solid_end_step_systems, thermal_end_step_systems);

  GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh_->entireBodyName());

  field_store->getParameterFields()[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return 100.0; });

  solid_system->setDisplacementBC(mesh_->domain("left"));
  thermal_system->setTemperatureBC(mesh_->domain("left"));

  solid_system->addTraction("right",
                            [=](double, auto X, auto, auto, auto, auto, auto temp, auto temperature_dot, auto) {
                              auto traction = 0.0 * X;
                              traction[0] = -0.015;
                              return traction;
                            });

  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics", coupled_cycle_zero_system, end_step_systems);

  // Run forward
  double dt = 1.0;
  for (int step = 0; step < 2; ++step) {
    physics->advanceTimestep(dt);
  }

  auto reactions = physics->getReactionStates();
  auto obj = 0.5 * (innerProduct(reactions[0], reactions[0]) + innerProduct(reactions[1], reactions[1]));

  gretl::set_as_objective(obj);
  obj.data_store().back_prop();

  auto param_sens = coupled->field_store->getParameterFields()[0].get_dual();
  EXPECT_TRUE(param_sens->Norml2() > 0.0);
}