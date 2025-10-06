// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>

#include "axom/slic.hpp"

#include "mfem.hpp"

#include "serac/numerics/solver_config.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac.hpp"

#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"

#include <cfenv>
#include <fem/datacollection.hpp>
#include <functional>
#include <mesh/vtk.hpp>
#include <set>
#include <string>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include <fenv.h>


int main(int argc, char* argv[])
 {

    //Initialize and automatically finalize MPI and other libraries
    serac::ApplicationManager applicationManager(argc, argv);

    // NOTE: p must be equal to 1 to work with Tribol's mortar method   
    constexpr int p = 1;

    //NOTE: dim must be equal to 2
    constexpr int dim = 2;

    //Create DataStore
    std::string name = "contact_ironing_2D_example";
    axom::sidre::DataStore datastore; 
    serac::StateManager::initialize(datastore, name + "_data");

    //Construct the appropiate dimension mesh and give it to the data store
    // std::string filename = SERAC_REPO_DIR "data/meshes/ironing_2D.mesh";
    // std::shared_ptr<serac::Mesh> mesh = std::make_shared<serac::Mesh>(filename, "ironing_2D_mesh", 2, 0);

    auto mesh = std::make_shared<serac::Mesh>(shared::MeshBuilder::Unify({
        shared::MeshBuilder::SquareMesh(4, 4).updateBdrAttrib(1, 6).updateBdrAttrib(3, 9).bdrAttribInfo().scale({1, 0.5}), 
        shared::MeshBuilder::SquareMesh(2, 2).scale({0.25, 0.25}).translate({0.0, 0.5}).updateBdrAttrib(3, 5).updateBdrAttrib(1,8).updateAttrib(1, 2)}), "iroing_2D_mesh", 0, 0);

    serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level=0};

    mfem::VisItDataCollection visit_dc("contact_ironing_visit", &mesh->mfemParMesh());

    visit_dc.SetPrefixPath("visit_out");
    visit_dc.Save();

    #ifndef MFEM_USE_STRUMPACK
        SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
        return 1;
    #endif

        serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::TrustRegion,
                                                        .relative_tol = 1.0e-8, 
                                                        .absolute_tol = 1.0e-10,
                                                        .max_iterations = 20,
                                                        .print_level = 1};

        serac::ContactOptions contact_options{.method = serac::ContactMethod::SmoothMortar,
                                              .enforcement = serac::ContactEnforcement::Penalty,
                                              .type = serac::ContactType::Frictionless,
                                              .penalty = 2000,
                                              .penalty2 = 0.0, 
                                              .jacobian = serac::ContactJacobian::Exact};

        serac::SolidMechanicsContact<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
            nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, mesh, 
            {"bulk_mod", "shear_mod"});


        serac::FiniteElementState K_field(serac::StateManager::newState(serac::L2<0>{}, "bulk_mod", mesh->tag()));

        mfem::Vector K_values({10.0, 100.0});
        mfem::PWConstCoefficient K_coeff(K_values);
        K_field.project(K_coeff);
        solid_solver.setParameter(0, K_field);

        serac::FiniteElementState G_field(serac::StateManager::newState(serac::L2<0>{}, "shear_mod", mesh->tag()));

        mfem::Vector G_values({0.1, 2.5});
        mfem::PWConstCoefficient G_coeff(G_values);
        G_field.project(G_coeff);
        solid_solver.setParameter(1, G_field);

        serac::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
        solid_solver.setMaterial(serac::DependsOn<0, 1>{}, mat, mesh->entireBody());

        //Pass the BC information to the solver object
        mesh->addDomainOfBoundaryElements("bottom_of_subtrate", serac::by_attr<dim>(6));
        solid_solver.setFixedBCs((mesh->domain("bottom_of_subtrate")));

        mesh->addDomainOfBoundaryElements("top of indenter", serac::by_attr<dim>(5));
        auto applied_displacement = [](serac::tensor<double, dim>, double t) {
            constexpr double init_steps = 10.0;
            serac::tensor<double, dim> u{};
            // std::cout << "T ========= " << t << std::endl;
            if (t <= init_steps + 1.0e-12) {
                u[1] = -t * 0.15 / init_steps;
                // std::cout << "In IF statement. u[1] = " << u[1] << " and t = " << t << std::endl;
            }
            else {
                u[0] = (t - init_steps) * 0.01;
                u[1] = -0.15;
                // std::cout << "in ELSE statement. u[1] = " << u[1] << " and u[0] = " << u[0] << " and t = " << t << std::endl;
            }
            return u;
        };

        solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("top of indenter"));
        // std::cout << "top of indenter size: " << mesh->domain("top of indenter").size() << std::endl;


        //Add the contact interaction
        auto contact_interaction_id = 0;
        std::set<int> surface_1_boundary_attributes({9});
        std::set<int> surface_2_boundary_attributes({8});
        solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes, surface_2_boundary_attributes, contact_options);

        //Finalize the data structures
        // solid_solver.completeSetup();

        std::string visit_name = name + "_visit";
        solid_solver.outputStateToDisk(visit_name);

        solid_solver.completeSetup();

        //Perform the quasi-static solve
        double dt = 1.0;

        for (int i{0}; i < 100; ++i) {
            solid_solver.advanceTimestep(dt);
            visit_dc.SetCycle(i);
            visit_dc.SetTime((i+1)*dt);
            visit_dc.Save();

            //Output the sidre-based plot files
            solid_solver.outputStateToDisk(visit_name);
        }

        return 0;
    }
    

        

    


