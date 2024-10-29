#include "mfem.hpp"

#include "serac/infrastructure/initialize.hpp"
#include <serac/infrastructure/terminator.hpp>
#include <serac/numerics/functional/domain.hpp>
#include <serac/numerics/functional/functional.hpp>
#include <serac/numerics/equation_solver.hpp>
#include <serac/numerics/solver_config.hpp>
#include <serac/numerics/stdfunction_operator.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/physics/boundary_conditions/boundary_condition_manager.hpp>
#include "serac/mesh/mesh_utils.hpp"


inline double matrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
    mfem::HypreParMatrix* H = K.get();
    hypre_ParCSRMatrix * Hhypre = static_cast<hypre_ParCSRMatrix *>(*H);
    double Hfronorm;
    hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
    return Hfronorm;
}


// _main_init_start
int main(int argc, char* argv[])
{
    // Initialize Serac
    serac::initialize(argc, argv);
    ::axom::sidre::DataStore datastore;
    ::serac::StateManager::initialize(datastore, "sidreDataStore");
    
    // Define the spatial dimension of the problem and the type of finite elements used.
    constexpr int ORDER {1};
    constexpr int DIM {2};

    constexpr double LENGTH = 8.0;
    constexpr double WIDTH = 1.0;
    constexpr int serial_refinement = 0;
    constexpr int parallel_refinement = 0;
    auto mesh = serac::mesh::refineAndDistribute(
        mfem::Mesh::MakeCartesian2D(8, 2, mfem::Element::TRIANGLE, true, LENGTH, WIDTH), 
        serial_refinement, parallel_refinement);
    std::string mesh_tag{"mesh"};
    auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

    using FunctionSpace = serac::H1<ORDER, DIM>;
    serac::FiniteElementState node_disp_computed(pmesh, FunctionSpace{}, "shape_displacement");
    node_disp_computed = 0.0;

    // Construct the new functional object using the known test and trial spaces
    serac::Functional<FunctionSpace(FunctionSpace)> residual(
        &node_disp_computed.space(), {&node_disp_computed.space()}); // shape, solution, and residual FESs

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

            auto stress = (1/serac::det(dy_dX))*serac::dot(P, serac::transpose(dy_dX));

            auto source = serac::zero{};
            return ::serac::tuple{source, stress};  /// N*source + DN*flux
        },
        pmesh
    );

    // It would be a good idea to put Dirichlet boundary conditions on this problem.
    // I would constrain each edge of the rectangle in the normal direction.
    // serac::BoundaryConditionManager bc_manager(pmesh);
    // left_disp_bdr_coef = std::make_shared<mfem::VectorFunctionCoefficient>(DIM, disp)
    // bc_manager.addEssential({0}, serac::GeneralCoefficient ess_bdr_coef,
    //                 mfem::ParFiniteElementSpace& space, const std::optional<int> component = {})
    mfem::Array<int> constrained_dofs;

    std::unique_ptr<mfem::HypreParMatrix> dresidualdu;

    // wrap residual and provide Jacobian
    serac::mfem_ext::StdFunctionOperator residual_opr(
        node_disp_computed.space().TrueVSize(),
        [&constrained_dofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
            double dummy_time = 1.0;
            const mfem::Vector res = residual(dummy_time, u);
            r = res;
            //r.SetSubVector(constrained_dofs, 0.0);
        },
        [&constrained_dofs, &residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
            double dummy_time = 1.0;
            auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
            dresidualdu       = assemble(dr_du);
            //dresidualdu->EliminateBC(constrained_dofs, mfem::Operator::DiagonalPolicy::DIAG_ONE);
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
    eq_solver.solve(node_disp_computed);

    // Check stiffness matrix symmetry
    double norm_K = matrixNorm(dresidualdu);
    std::unique_ptr<mfem::HypreParMatrix> K_skew = std::make_unique<mfem::HypreParMatrix>(*dresidualdu);
    K_skew->Add(-1.0, *(dresidualdu->Transpose()));
    *K_skew *= 0.5;
    double norm_K_skew = matrixNorm(K_skew);
    std::cout << "K norm         = " << norm_K << "\n"
              << "skew part norm = " << norm_K_skew << std::endl;

    mfem::ParGridFunction nodeSolGF(&node_disp_computed.space());
    nodeSolGF.SetFromTrueDofs(node_disp_computed);
    nodeSolGF.Print();
    auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D", &pmesh);
    pd.RegisterField("displacement", &nodeSolGF);
    pd.SetCycle(1);
    pd.SetTime(1);
    pd.Save();

    return 0;
}