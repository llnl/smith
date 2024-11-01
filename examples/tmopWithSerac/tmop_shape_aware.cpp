#include "mfem.hpp"

#include "serac/infrastructure/initialize.hpp"
#include <serac/infrastructure/terminator.hpp>
#include <serac/numerics/functional/domain.hpp>
#include <serac/numerics/functional/functional.hpp>
#include <serac/numerics/functional/shape_aware_functional.hpp>
#include <serac/numerics/equation_solver.hpp>
#include <serac/numerics/solver_config.hpp>
#include <serac/numerics/stdfunction_operator.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/physics/boundary_conditions/boundary_condition_manager.hpp>
#include "serac/mesh/mesh_utils.hpp"

#include <functional>


/**
 * Compute Frobenius norm of an mfem HypreParMatrix
 */
inline double matrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
    mfem::HypreParMatrix* H = K.get();
    hypre_ParCSRMatrix * Hhypre = static_cast<hypre_ParCSRMatrix *>(*H);
    double Hfronorm;
    hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
    return Hfronorm;
}


int main(int argc, char* argv[])
{
    // Initialize Serac
    serac::initialize(argc, argv);
    ::axom::sidre::DataStore datastore;
    ::serac::StateManager::initialize(datastore, "sidreDataStore");
    
    // Define the spatial dimension of the problem and the type of finite elements used.
    constexpr int ORDER {1};
    constexpr int DIM {2};

    // create mesh
    constexpr double LENGTH = 8.0;
    constexpr double WIDTH = 8.0;
    constexpr int serial_refinement = 0;
    constexpr int parallel_refinement = 0;
    auto mesh = serac::mesh::refineAndDistribute(
        mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::TRIANGLE, true, LENGTH, WIDTH), 
        serial_refinement, parallel_refinement);
    std::string mesh_tag{"mesh"};
    auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

    // instantiate solution field
    using FunctionSpace = serac::H1<ORDER, DIM>;
    serac::FiniteElementState mesh_displacement(pmesh, FunctionSpace{}, "mesh_displacement");
    mesh_displacement = 0.0;

    serac::FiniteElementState physical_displacement(pmesh, FunctionSpace{}, "displacement");
    //physical_displacement = 0.0;
    physical_displacement.Randomize(0);
    physical_displacement *= 0.25/physical_displacement.Max();

    // Construct the new functional object using the known test and trial spaces
    serac::ShapeAwareFunctional<FunctionSpace, FunctionSpace(FunctionSpace)> residual(
        &physical_displacement.space(), &mesh_displacement.space(), {&mesh_displacement.space()});

    // Define residual of problem
    residual.AddDomainIntegral(serac::Dimension<DIM>{}, serac::DependsOn<0>{},
        [=](double /*t*/, auto position, auto displacement) {
            auto du_dy = serac::get<1>(displacement);
            auto dx_dy = serac::DenseIdentity<DIM>() + du_dy;

            // triangular correction
            auto [_, dy_dxi] = position;
            serac::mat2 dX_dxi{{{1.0, 0.5},
                                {0.0, 0.5*std::sqrt(3)}}};
            
            serac::mat2 dxi_dX = serac::inv(dX_dxi);
            auto dy_dX = serac::dot(dy_dxi, dxi_dX);
            auto F = serac::dot(dx_dy, dy_dX);
            
            auto detF = serac::det(F);
            auto invFT = serac::transpose(serac::inv(F));
            auto P = 1/(detF)*(F - 0.5*serac::inner(F, F)*invFT);

            // Push stress forward from initial configuration to relaxed
            auto stress = (1/serac::det(dy_dX))*serac::dot(P, serac::transpose(dy_dX));

            auto source = serac::zero{};
            return ::serac::tuple{source, stress};  /// N*source + DN*flux
        },
        pmesh
    );

    // Dirichlet bondary conditions
    serac::BoundaryConditionManager bc_manager(pmesh);
    // constrain displacements normal to boundary
    auto zero_function = [](const mfem::Vector&) {
        return 0.0;
    };
    auto ess_bdr_coef = std::make_shared<mfem::FunctionCoefficient>(zero_function);
    bc_manager.addEssential({1, 3}, ess_bdr_coef, mesh_displacement.space(), 1);
    bc_manager.addEssential({2, 4}, ess_bdr_coef, mesh_displacement.space(), 0);
    mfem::Array<int> constrained_dofs = bc_manager.allEssentialTrueDofs();

    // impose the boundary conditions on the physical displacement
    physical_displacement.SetSubVector(constrained_dofs, 0.0);

    std::unique_ptr<mfem::HypreParMatrix> dresidualdu;

    // wrap residual and provide Jacobian
    serac::mfem_ext::StdFunctionOperator residual_opr(
        mesh_displacement.space().TrueVSize(),
        [&residual, &physical_displacement, &constrained_dofs](const mfem::Vector& u, mfem::Vector& r) {
            double dummy_time = 1.0;
            const mfem::Vector res = residual(dummy_time, physical_displacement, u);
            r = res;
            r.SetSubVector(constrained_dofs, 0.0);

        },
        [&residual, &physical_displacement, &constrained_dofs, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
            double dummy_time = 1.0;
            auto [val, dr_du] = residual(dummy_time, physical_displacement, serac::differentiate_wrt(u));
            dresidualdu       = assemble(dr_du);
            dresidualdu->EliminateBC(constrained_dofs, mfem::Operator::DiagonalPolicy::DIAG_ONE);
            return *dresidualdu;
        }
    );

    const serac::LinearSolverOptions lin_opts = {
        .linear_solver = ::serac::LinearSolver::CG,
        //.linear_solver  = serac::LinearSolver::Strumpack,
        .preconditioner = ::serac::Preconditioner::HypreJacobi,
        .relative_tol   = 1.0e-10,
        .absolute_tol   = 1.0e-12,
        .max_iterations = 100,
        .print_level    = 0
    };

    const serac::NonlinearSolverOptions nonlin_opts = {
        //.nonlin_solver = ::serac::NonlinearSolver::Newton,
        .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
        // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
        .relative_tol   = 1.0e-8,
        .absolute_tol   = 1.0e-10,
        // .min_iterations = 1, 
        .max_iterations = 1000, // 2000
        // .max_line_search_iterations = 20, //0
        .print_level    = 1
    };

    serac::EquationSolver eq_solver(nonlin_opts, lin_opts, pmesh.GetComm());
    eq_solver.setOperator(residual_opr);
    eq_solver.solve(mesh_displacement);

    // Check stiffness matrix symmetry
    double norm_K = matrixNorm(dresidualdu);
    std::unique_ptr<mfem::HypreParMatrix> K_skew = std::make_unique<mfem::HypreParMatrix>(*dresidualdu);
    K_skew->Add(-1.0, *(dresidualdu->Transpose()));
    *K_skew *= 0.5;
    double norm_K_skew = matrixNorm(K_skew);
    std::cout << "K norm         = " << norm_K << "\n"
              << "skew part norm = " << norm_K_skew << std::endl;

    mfem::ParGridFunction nodeSolGF(&mesh_displacement.space());
    nodeSolGF.SetFromTrueDofs(mesh_displacement);
    nodeSolGF.Print();
    mfem::ParGridFunction physical_displacement_GF(&physical_displacement.space());
    physical_displacement_GF.SetFromTrueDofs(physical_displacement);
    auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D", &pmesh);
    pd.RegisterField("mesh_displacement", &nodeSolGF);
    pd.RegisterField("physical_displacement", &physical_displacement_GF);
    pd.SetCycle(1);
    pd.SetTime(1);
    pd.Save();

    return 0;
}