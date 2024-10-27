#include "mfem.hpp"

#include "serac/infrastructure/initialize.hpp"
#include <serac/infrastructure/terminator.hpp>
#include <serac/numerics/functional/domain.hpp>
#include <serac/numerics/functional/functional.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/numerics/equation_solver.hpp>
#include <serac/numerics/solver_config.hpp>
#include <serac/numerics/stdfunction_operator.hpp>
#include "serac/mesh/mesh_utils.hpp"


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
            // x = X + u
            auto [X, dXdxi] = position;
            auto du_dX = serac::get<1>(displacement);

            // auto mu = 0.5 * (serac::inner(Tmat, Tmat) / abs(serac::det(Tmat))) - 1.0;
            // triangular correction = [ 1, -1/sqrt(3); 0, 2/sqrt(3)]
            serac::mat2 WInvMat = {{{1.0, -0.5773502691896258}, 
                                    {0.0,  1.1547005383792517}}};
            // serac::mat2 WInvMat = serac::DenseIdentity<2>();

            // Jacobian from parent element to the physical space (i.e., dx_dxi)
            auto Amat = dXdxi + serac::dot(du_dX, dXdxi);

            // Target matrix (updated Jacobian, Tmat or T)
            auto T = serac::dot(Amat, WInvMat);

            auto detT = serac::det(T);
            auto B = serac::dot(T, serac::transpose(T));
            auto stress = 1/(detT*detT)*(B - 0.5*serac::inner(T, T)*serac::DenseIdentity<DIM>());

            auto source = serac::zero{};
            return ::serac::tuple{source, stress};  /// N*source + DN*flux
        },
        pmesh
    );

    // It would be a good idea to put Dirichlet boundary conditions on this problem.
    // I would constrain each edge of the rectangle in the normal direction.
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
        .nonlin_solver = ::serac::NonlinearSolver::Newton,
        // .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
        // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
        .relative_tol   = 1.0e-8,
        .absolute_tol   = 1.0e-10,
        // .min_iterations = 1, 
        .max_iterations = 20, // 2000
        // .max_line_search_iterations = 20, //0
        .print_level    = 1
    };

    serac::EquationSolver eq_solver(nonlin_opts, lin_opts, pmesh.GetComm());
    eq_solver.setOperator(residual_opr);
    eq_solver.solve(node_disp_computed);

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