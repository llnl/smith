// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>

#include <set>
#include <string>

#include "axom/slic.hpp"

#include "mfem.hpp"

#include "smith/smith.hpp"

// this example is intended to eventually replace twist.cpp
int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  using VectorSpace = smith::H1<p, dim>;

  // Create DataStore
  std::string name = "contact_twist_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/twohex_for_contact.mesh";
  auto mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), "twist_mesh", 3, 0);

  mesh->addDomainOfBoundaryElements("fixed_surface", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("driven_surface", smith::by_attr<dim>(6));

  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::LagrangeMultiplier,
                                        .type = smith::ContactType::Frictionless,
                                        .jacobian = smith::ContactJacobian::Exact};

  std::string contact_constraint_name = "default_contact";

  // Specify the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({4});
  std::set<int> surface_2_boundary_attributes({5});
  smith::ContactConstraint contact_constraint(contact_interaction_id, mesh->mfemParMesh(),
                                              surface_1_boundary_attributes, surface_2_boundary_attributes,
                                              contact_options, contact_constraint_name);

  smith::FiniteElementState shape = smith::StateManager::newState(VectorSpace{}, "shape", mesh->tag());
  smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());

  std::vector<smith::FiniteElementState> contact_states;
  contact_states = {shape, disp};
  // initialize displacement
  contact_states[smith::ContactFields::DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
    auto u = 0.1 * x;
    return u;
  });

  contact_states[smith::ContactFields::SHAPE] = 0.0;

  double time = 0.0, dt = 1.0;
  int direction = 0;
  auto input_states = getConstFieldPointers(contact_states);
  auto gap = contact_constraint.evaluate(time, dt, input_states);
  auto gap_Jacobian = contact_constraint.jacobian(time, dt, input_states, direction);

  return 0;
}
