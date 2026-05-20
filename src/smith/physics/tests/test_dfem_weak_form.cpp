// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/dfem_solid_weak_form.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/tests/physics_test_utils.hpp"

namespace mfem {
namespace future {

template <typename T, int dim>
MFEM_HOST_DEVICE auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

}  // namespace future
}  // namespace mfem

namespace dfem_weak_form_test {

struct VectorDiffusionDiagnosticQf {
  static constexpr int dim = 2;

  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& dudxi,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>& J,
                                           const mfem::real_t& w,
                                           mfem::future::tensor<mfem::real_t, dim, dim>& dvdxi) const
  {
    const auto invJ = mfem::future::inv(J);
    const auto invJt = mfem::future::transpose(invJ);
    dvdxi = (dudxi * invJ) * invJt * mfem::future::det(J) * w;
  }
};

struct SolidUCoordsQf {
  static constexpr int dim = 2;
  double K;
  double G;

  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi,
                                           mfem::real_t weight,
                                           mfem::future::tensor<mfem::real_t, dim, dim>& out) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    const auto zero = mfem::future::tensor<mfem::real_t, dim, dim>{};
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    auto P = mfem::future::dot(F, S);
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    out = -P * JxW;
  }
};

struct SolidUVCoordsQf {
  static constexpr int dim = 2;
  double K;
  double G;

  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>&,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi,
                                           mfem::real_t weight,
                                           mfem::future::tensor<mfem::real_t, dim, dim>& out) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    const auto zero = mfem::future::tensor<mfem::real_t, dim, dim>{};
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    auto P = mfem::future::dot(F, S);
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    out = -P * JxW;
  }
};

struct SolidUVACoordsQf {
  static constexpr int dim = 2;
  double K;
  double G;

  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>&,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>&,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi,
                                           mfem::real_t weight,
                                           mfem::future::tensor<mfem::real_t, dim, dim>& out) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    const auto zero = mfem::future::tensor<mfem::real_t, dim, dim>{};
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    auto P = mfem::future::dot(F, S);
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    out = -P * JxW;
  }
};

struct SolidUVACoordsDensityQf {
  static constexpr int dim = 2;
  double K;
  double G;

  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>&,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>&,
                                           const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi,
                                           mfem::real_t weight, mfem::real_t,
                                           mfem::future::tensor<mfem::real_t, dim, dim>& out) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    const auto zero = mfem::future::tensor<mfem::real_t, dim, dim>{};
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    auto P = mfem::future::dot(F, S);
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    out = -P * JxW;
  }
};

}  // namespace dfem_weak_form_test

namespace {

auto element_shape = mfem::Element::QUADRILATERAL;

struct StVenantKirchhoffWithFieldDensityDfem {
  static constexpr int dim = 2;

  template <typename T, int dim_, typename Density>
  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<T, dim_, dim_>& du_dX,
                                  const mfem::future::tensor<T, dim_, dim_>&, const Density&) const
  {
    auto I = mfem::future::IdentityMatrix<dim_>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    return mfem::future::tuple{mfem::future::dot(F, S)};
  }

  template <typename Density>
  SMITH_HOST_DEVICE auto density(const Density& density) const
  {
    return density;
  }

  double K;
  double G;
};

struct DfemWeakFormFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;
  using SolidT = smith::DfemSolidWeakForm;

  enum FIELD
  {
    DISP = SolidT::DISPLACEMENT,
    VELO = SolidT::VELOCITY,
    ACCEL = SolidT::ACCELERATION,
    COORDS = SolidT::COORDINATES,
    DENSITY = SolidT::NUM_STATE_VARS
  };

  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "dfem_weak_form");

    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian2D(4, 3, element_shape, true, 1.0, 0.75),
                                         "dfem_mesh", 0, 0);

    disp = std::make_unique<smith::FiniteElementState>(smith::StateManager::newState(VectorSpace{}, "disp", mesh->tag()));
    velo = std::make_unique<smith::FiniteElementState>(smith::StateManager::newState(VectorSpace{}, "velo", mesh->tag()));
    accel =
        std::make_unique<smith::FiniteElementState>(smith::StateManager::newState(VectorSpace{}, "accel", mesh->tag()));
    density =
        std::make_unique<smith::FiniteElementState>(smith::StateManager::newState(DensitySpace{}, "density", mesh->tag()));

    coords = std::make_unique<smith::FiniteElementState>(
        *static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(), "coordinates");
    coords->setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));

    disp_tangent = std::make_unique<smith::FiniteElementState>(disp->space(), "disp_tangent");
    density_tangent = std::make_unique<smith::FiniteElementState>(density->space(), "density_tangent");
    pseudoRand(*disp_tangent);
    pseudoRand(*density_tangent);

    disp->setFromFieldFunction([](smith::tensor<double, dim> x) {
      smith::tensor<double, dim> u({0.02 * x[0], -0.03 * x[1]});
      return u;
    });
    *velo = 0.0;
    *accel = 0.0;
    *density = 1.4;

    shape_disp = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());
    shape_disp_dual = std::make_unique<smith::FiniteElementDual>(mesh->newShapeDisplacementDual());

    weak_form = std::make_shared<SolidT>("solid", mesh, disp->space(), std::vector<const mfem::ParFiniteElementSpace*>{&density->space()});

    StVenantKirchhoffWithFieldDensityDfem mat;
    double E = 1.0e3;
    double nu = 0.3;
    mat.K = E / (3.0 * (1.0 - 2.0 * nu));
    mat.G = E / (2.0 * (1.0 + nu));

    const mfem::IntegrationRule& ir = mfem::IntRules.Get(disp->space().GetFE(0)->GetGeomType(), 2);
    weak_form->setMaterial<StVenantKirchhoffWithFieldDensityDfem, smith::ScalarParameter<0>>(
        mesh->mfemParMesh().attributes, mat, ir);

  }

  std::vector<smith::ConstFieldPtr> inputFields() const
  {
    return {disp.get(), velo.get(), accel.get(), coords.get(), density.get()};
  }

  std::vector<double> oneHotWeights(size_t active_field) const
  {
    std::vector<double> weights(inputFields().size(), 0.0);
    weights[active_field] = 1.0;
    return weights;
  }

  StVenantKirchhoffWithFieldDensityDfem materialModel() const
  {
    StVenantKirchhoffWithFieldDensityDfem mat;
    double E = 1.0e3;
    double nu = 0.3;
    mat.K = E / (3.0 * (1.0 - 2.0 * nu));
    mat.G = E / (2.0 * (1.0 + nu));
    return mat;
  }

  mfem::Vector fieldVector(int field_id, const smith::FiniteElementState* disp_override = nullptr) const
  {
    switch (field_id) {
      case DISP:
        return mfem::Vector(disp_override ? static_cast<const mfem::Vector&>(*disp_override)
                                          : static_cast<const mfem::Vector&>(*disp));
      case VELO:
        return mfem::Vector(*velo);
      case ACCEL:
        return mfem::Vector(*accel);
      case COORDS:
        return mfem::Vector(*coords);
      case DENSITY:
        return mfem::Vector(*density);
      default:
        SLIC_ERROR("unexpected field id");
        return mfem::Vector();
    }
  }

  template <typename Qf, typename InputTuple, size_t... DerivIds>
  double derivativeActionFiniteDifferenceMismatch(Qf& qf, InputTuple inputs, std::vector<mfem::future::FieldDescriptor> infds,
                                                  std::index_sequence<DerivIds...>, double eps = 1.0e-7) const
  {
    mfem::future::DifferentiableOperator dop(infds, {{DISP, &disp->space()}}, mesh->mfemParMesh());
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(disp->space().GetFE(0)->GetGeomType(), 2);
    dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
        qf, inputs, mfem::future::tuple{mfem::future::Gradient<DISP>{}}, ir, mesh->mfemParMesh().attributes,
        std::index_sequence<DerivIds...>{});

    std::vector<mfem::Vector> base_fields;
    base_fields.reserve(infds.size());
    for (const auto& fd : infds) {
      base_fields.push_back(fieldVector(static_cast<int>(fd.id)));
    }
    mfem::MultiVector X(static_cast<int>(base_fields.size()));
    for (int i = 0; i < static_cast<int>(base_fields.size()); ++i) {
      X.MakeRef(i, base_fields[i]);
    }

    auto dRdU = dop.GetDerivative(DISP, X);
    mfem::Vector action_t(disp->space().GetTrueVSize());
    action_t = 0.0;
    mfem::MultiVector Z{action_t};
    dRdU->Mult(*disp_tangent, Z);

    smith::FiniteElementState disp_plus(*disp);
    smith::FiniteElementState disp_minus(*disp);
    disp_plus.Add(eps, *disp_tangent);
    disp_minus.Add(-eps, *disp_tangent);

    std::vector<mfem::Vector> plus_fields;
    std::vector<mfem::Vector> minus_fields;
    plus_fields.reserve(infds.size());
    minus_fields.reserve(infds.size());
    for (const auto& fd : infds) {
      plus_fields.push_back(fieldVector(static_cast<int>(fd.id), &disp_plus));
      minus_fields.push_back(fieldVector(static_cast<int>(fd.id), &disp_minus));
    }

    mfem::MultiVector X_plus(static_cast<int>(plus_fields.size()));
    mfem::MultiVector X_minus(static_cast<int>(minus_fields.size()));
    for (int i = 0; i < static_cast<int>(plus_fields.size()); ++i) {
      X_plus.MakeRef(i, plus_fields[i]);
      X_minus.MakeRef(i, minus_fields[i]);
    }

    mfem::Vector r_plus(disp->space().GetTrueVSize());
    mfem::Vector r_minus(disp->space().GetTrueVSize());
    mfem::MultiVector R_plus{r_plus};
    mfem::MultiVector R_minus{r_minus};
    dop.Mult(X_plus, R_plus);
    dop.Mult(X_minus, R_minus);

    r_plus -= r_minus;
    r_plus *= 0.5 / eps;
    r_plus -= action_t;
    return r_plus.Norml2();
  }

  smith::FiniteElementDual finiteDifferenceAction(size_t active_field, const smith::FiniteElementState& tangent,
                                                  double eps = 1.0e-7) const
  {
    smith::FiniteElementState disp_plus(*disp);
    smith::FiniteElementState density_plus(*density);
    smith::FiniteElementState disp_minus(*disp);
    smith::FiniteElementState density_minus(*density);

    auto perturb = [&](smith::FiniteElementState& plus_field, smith::FiniteElementState& minus_field) {
      plus_field.Add(eps, tangent);
      minus_field.Add(-eps, tangent);
    };

    switch (active_field) {
      case DISP:
        perturb(disp_plus, disp_minus);
        break;
      case DENSITY:
        perturb(density_plus, density_minus);
        break;
      default:
        ADD_FAILURE() << "unexpected active_field";
        return smith::FiniteElementDual(disp->space(), "fd_invalid");
    }

    smith::FiniteElementDual fd(disp->space(), "fd");
    auto r_plus = weak_form->residual(time_info, shape_disp.get(), {&disp_plus, velo.get(), accel.get(), coords.get(), &density_plus});
    auto r_minus =
        weak_form->residual(time_info, shape_disp.get(), {&disp_minus, velo.get(), accel.get(), coords.get(), &density_minus});
    fd = r_plus;
    fd -= r_minus;
    fd *= 0.5 / eps;
    return fd;
  }

  const smith::TimeInfo time_info{0.0, 1.0};

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<SolidT> weak_form;

  std::unique_ptr<smith::FiniteElementState> disp;
  std::unique_ptr<smith::FiniteElementState> velo;
  std::unique_ptr<smith::FiniteElementState> accel;
  std::unique_ptr<smith::FiniteElementState> coords;
  std::unique_ptr<smith::FiniteElementState> density;

  std::unique_ptr<smith::FiniteElementState> disp_tangent;
  std::unique_ptr<smith::FiniteElementState> density_tangent;

  std::unique_ptr<smith::FiniteElementState> shape_disp;
  std::unique_ptr<smith::FiniteElementDual> shape_disp_dual;
};

TEST_F(DfemWeakFormFixture, JacobianMatchesFiniteDifferenceForDisplacement)
{
  auto J = weak_form->jacobian(time_info, shape_disp.get(), inputFields(), oneHotWeights(DISP));
  smith::FiniteElementDual jac_times_dir(disp->space(), "jac_times_dir");
  J->Mult(*disp_tangent, jac_times_dir);

  auto fd = finiteDifferenceAction(DISP, *disp_tangent);
  jac_times_dir -= fd;
  EXPECT_NEAR(jac_times_dir.Norml2(), 0.0, 1.0e-7);
}

TEST_F(DfemWeakFormFixture, JacobianMatchesFiniteDifferenceForDensity)
{
  auto J = weak_form->jacobian(time_info, shape_disp.get(), inputFields(), oneHotWeights(DENSITY));
  smith::FiniteElementDual jac_times_dir(disp->space(), "jac_times_dir");
  J->Mult(*density_tangent, jac_times_dir);

  auto fd = finiteDifferenceAction(DENSITY, *density_tangent);
  jac_times_dir -= fd;
  EXPECT_NEAR(jac_times_dir.Norml2(), 0.0, 1.0e-8);
}

TEST_F(DfemWeakFormFixture, VjpMatchesJacobianTransposeProducts)
{
  smith::FiniteElementState v(disp->space(), "seed");
  pseudoRand(v);

  smith::FiniteElementDual disp_vjp_expected(disp->space(), "disp_vjp_expected");
  smith::FiniteElementDual density_vjp_expected(density->space(), "density_vjp_expected");

  auto J_disp = weak_form->jacobian(time_info, shape_disp.get(), inputFields(), oneHotWeights(DISP));
  J_disp->AddMultTranspose(v, disp_vjp_expected);

  auto J_density = weak_form->jacobian(time_info, shape_disp.get(), inputFields(), oneHotWeights(DENSITY));
  J_density->AddMultTranspose(v, density_vjp_expected);

  std::vector<smith::FiniteElementDual> field_vjps;
  field_vjps.emplace_back(disp->space(), "disp_vjp");
  field_vjps.emplace_back(velo->space(), "velo_vjp");
  field_vjps.emplace_back(accel->space(), "accel_vjp");
  field_vjps.emplace_back(coords->space(), "coords_vjp");
  field_vjps.emplace_back(density->space(), "density_vjp");

  auto field_vjp_ptrs = smith::getFieldPointers(field_vjps);
  weak_form->vjp(time_info, shape_disp.get(), inputFields(), {}, &v, shape_disp_dual.get(), field_vjp_ptrs, {});

  (*field_vjp_ptrs[DISP]) -= disp_vjp_expected;
  (*field_vjp_ptrs[DENSITY]) -= density_vjp_expected;

  EXPECT_NEAR(field_vjp_ptrs[DISP]->Norml2(), 0.0, 1.0e-7);
  EXPECT_NEAR(field_vjp_ptrs[VELO]->Norml2(), 0.0, 1.0e-12);
  EXPECT_NEAR(field_vjp_ptrs[ACCEL]->Norml2(), 0.0, 1.0e-12);
  EXPECT_NEAR(field_vjp_ptrs[COORDS]->Norml2(), 0.0, 1.0e-12);
  EXPECT_NEAR(field_vjp_ptrs[DENSITY]->Norml2(), 0.0, 1.0e-7);
}

TEST_F(DfemWeakFormFixture, DiagnosticVectorDiffusionAssemblyOnSmithOrdering)
{
  if constexpr (!mfem::future::LocalQFBackend::has_cached_derivative) {
    GTEST_SKIP() << "LocalQFBackend sparse assembly unavailable with cached derivatives disabled";
  }

  auto& pfes = disp->space();
  auto* nodes = static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes());
  auto* mfes = nodes->ParFESpace();

  mfem::ParBilinearForm blf(&pfes);
  blf.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator);
  blf.SetAssemblyLevel(mfem::AssemblyLevel::LEGACYFULL);
  blf.Assemble();
  blf.Finalize();

  static constexpr int U = 0;
  static constexpr int COORDS = 1;

  mfem::future::DifferentiableOperator dop({{U, &pfes}, {COORDS, mfes}}, {{U, &pfes}}, mesh->mfemParMesh());
  dfem_weak_form_test::VectorDiffusionDiagnosticQf qf;

  const mfem::IntegrationRule& ir = mfem::IntRules.Get(pfes.GetFE(0)->GetGeomType(), 2 * disp_order);
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Gradient<U>{}, mfem::future::Gradient<COORDS>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<U>{}}, ir, mesh->mfemParMesh().attributes,
      std::integer_sequence<size_t, U>{});

  mfem::Vector x_t(*disp_tangent);
  mfem::Vector nodes_t;
  nodes->GetTrueDofs(nodes_t);
  mfem::MultiVector X{x_t, nodes_t};
  auto dRdU = dop.GetDerivative(U, X);

  mfem::SparseMatrix* A = nullptr;
  dRdU->Assemble(A);

  mfem::ParGridFunction x_local(&pfes);
  x_local = 0.0;
  disp_tangent->fillGridFunction(x_local);

  mfem::ParGridFunction y_ref(&pfes);
  y_ref = 0.0;
  blf.Mult(x_local, y_ref);

  mfem::ParGridFunction y_dfem(&pfes);
  y_dfem = 0.0;
  A->Mult(x_local, y_dfem);

  y_dfem -= y_ref;
  EXPECT_NEAR(y_dfem.Norml2(), 0.0, 1.0e-10);

  delete A;
}

TEST_F(DfemWeakFormFixture, DiagnosticSolidDerivativeActionMatchesAssembledSparse)
{
  if constexpr (!mfem::future::LocalQFBackend::has_cached_derivative) {
    GTEST_SKIP() << "LocalQFBackend sparse assembly unavailable with cached derivatives disabled";
  }

  auto& pfes = disp->space();
  auto& density_fes = density->space();
  auto* nodes = static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes());
  auto* mfes = nodes->ParFESpace();

  static constexpr int DISP_ID = SolidT::DISPLACEMENT;
  static constexpr int VELO_ID = SolidT::VELOCITY;
  static constexpr int ACCEL_ID = SolidT::ACCELERATION;
  static constexpr int COORDS_ID = SolidT::COORDINATES;
  static constexpr int DENSITY_ID = SolidT::NUM_STATE_VARS;

  mfem::future::DifferentiableOperator dop({{DISP_ID, &pfes},
                                            {VELO_ID, &pfes},
                                            {ACCEL_ID, &pfes},
                                            {COORDS_ID, mfes},
                                            {DENSITY_ID, &density_fes}},
                                           {{DISP_ID, &pfes}}, mesh->mfemParMesh());

  StVenantKirchhoffWithFieldDensityDfem mat;
  double E = 1.0e3;
  double nu = 0.3;
  mat.K = E / (3.0 * (1.0 - 2.0 * nu));
  mat.G = E / (2.0 * (1.0 + nu));

  smith::StressDivQFunction<StVenantKirchhoffWithFieldDensityDfem, smith::ScalarParameter<0>> qf{.material = mat};
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(pfes.GetFE(0)->GetGeomType(), 2);
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Gradient<DISP_ID>{}, mfem::future::Gradient<VELO_ID>{},
                          mfem::future::Gradient<ACCEL_ID>{}, mfem::future::Gradient<COORDS_ID>{},
                          mfem::future::Weight{}, mfem::future::Value<DENSITY_ID>{}},
      mfem::future::tuple{mfem::future::Gradient<DISP_ID>{}}, ir, mesh->mfemParMesh().attributes,
      std::integer_sequence<size_t, DISP_ID, DENSITY_ID>{});

  mfem::Vector disp_t(*disp);
  mfem::Vector velo_t(*velo);
  mfem::Vector accel_t(*accel);
  mfem::Vector coords_t(*coords);
  mfem::Vector density_t(*density);
  mfem::MultiVector X{disp_t, velo_t, accel_t, coords_t, density_t};

  auto dRdU = dop.GetDerivative(DISP_ID, X);

  mfem::Vector action_t(pfes.GetTrueVSize());
  action_t = 0.0;
  mfem::MultiVector Z{action_t};
  dRdU->Mult(*disp_tangent, Z);

  mfem::SparseMatrix* A = nullptr;
  dRdU->Assemble(A);

  mfem::ParGridFunction x_local(&pfes);
  x_local = 0.0;
  disp_tangent->fillGridFunction(x_local);

  mfem::ParGridFunction y_local(&pfes);
  y_local = 0.0;
  A->Mult(x_local, y_local);

  mfem::Vector assembled_t(pfes.GetTrueVSize());
  pfes.GetProlongationMatrix()->MultTranspose(y_local, assembled_t);

  assembled_t -= action_t;
  EXPECT_NEAR(assembled_t.Norml2(), 0.0, 1.0e-10);

  delete A;
}

TEST_F(DfemWeakFormFixture, DiagnosticSolidDerivativeActionMatchesFiniteDifference)
{
  auto& pfes = disp->space();
  auto& density_fes = density->space();
  auto* nodes = static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes());
  auto* mfes = nodes->ParFESpace();

  static constexpr int DISP_ID = SolidT::DISPLACEMENT;
  static constexpr int VELO_ID = SolidT::VELOCITY;
  static constexpr int ACCEL_ID = SolidT::ACCELERATION;
  static constexpr int COORDS_ID = SolidT::COORDINATES;
  static constexpr int DENSITY_ID = SolidT::NUM_STATE_VARS;

  mfem::future::DifferentiableOperator dop({{DISP_ID, &pfes},
                                            {VELO_ID, &pfes},
                                            {ACCEL_ID, &pfes},
                                            {COORDS_ID, mfes},
                                            {DENSITY_ID, &density_fes}},
                                           {{DISP_ID, &pfes}}, mesh->mfemParMesh());

  StVenantKirchhoffWithFieldDensityDfem mat;
  double E = 1.0e3;
  double nu = 0.3;
  mat.K = E / (3.0 * (1.0 - 2.0 * nu));
  mat.G = E / (2.0 * (1.0 + nu));

  smith::StressDivQFunction<StVenantKirchhoffWithFieldDensityDfem, smith::ScalarParameter<0>> qf{.material = mat};
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(pfes.GetFE(0)->GetGeomType(), 2);
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Gradient<DISP_ID>{}, mfem::future::Gradient<VELO_ID>{},
                          mfem::future::Gradient<ACCEL_ID>{}, mfem::future::Gradient<COORDS_ID>{},
                          mfem::future::Weight{}, mfem::future::Value<DENSITY_ID>{}},
      mfem::future::tuple{mfem::future::Gradient<DISP_ID>{}}, ir, mesh->mfemParMesh().attributes,
      std::integer_sequence<size_t, DISP_ID, DENSITY_ID>{});

  mfem::Vector disp_t(*disp);
  mfem::Vector velo_t(*velo);
  mfem::Vector accel_t(*accel);
  mfem::Vector coords_t(*coords);
  mfem::Vector density_t(*density);
  mfem::MultiVector X{disp_t, velo_t, accel_t, coords_t, density_t};

  auto dRdU = dop.GetDerivative(DISP_ID, X);

  mfem::Vector action_t(pfes.GetTrueVSize());
  action_t = 0.0;
  mfem::MultiVector Z{action_t};
  dRdU->Mult(*disp_tangent, Z);

  constexpr double eps = 1.0e-7;
  mfem::Vector disp_plus(*disp);
  mfem::Vector disp_minus(*disp);
  disp_plus.Add(eps, *disp_tangent);
  disp_minus.Add(-eps, *disp_tangent);

  mfem::MultiVector X_plus{disp_plus, velo_t, accel_t, coords_t, density_t};
  mfem::MultiVector X_minus{disp_minus, velo_t, accel_t, coords_t, density_t};

  mfem::Vector r_plus(pfes.GetTrueVSize());
  mfem::Vector r_minus(pfes.GetTrueVSize());
  mfem::MultiVector R_plus{r_plus};
  mfem::MultiVector R_minus{r_minus};
  dop.Mult(X_plus, R_plus);
  dop.Mult(X_minus, R_minus);

  r_plus -= r_minus;
  r_plus *= 0.5 / eps;
  r_plus -= action_t;

  EXPECT_NEAR(r_plus.Norml2(), 0.0, 1.0e-7);
}

TEST_F(DfemWeakFormFixture, MinimalSolidDerivativeStages)
{
  auto* nodes = static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes());
  auto* mfes = nodes->ParFESpace();
  auto mat = materialModel();

  {
    dfem_weak_form_test::SolidUCoordsQf qf{.K = mat.K, .G = mat.G};
    auto mismatch = derivativeActionFiniteDifferenceMismatch(
        qf, mfem::future::tuple{mfem::future::Gradient<DISP>{}, mfem::future::Gradient<COORDS>{}, mfem::future::Weight{}},
        {{DISP, &disp->space()}, {COORDS, mfes}}, std::index_sequence<DISP>{});
    EXPECT_NEAR(mismatch, 0.0, 1.0e-7);
  }

  {
    dfem_weak_form_test::SolidUVCoordsQf qf{.K = mat.K, .G = mat.G};
    auto mismatch = derivativeActionFiniteDifferenceMismatch(
        qf,
        mfem::future::tuple{mfem::future::Gradient<DISP>{}, mfem::future::Gradient<VELO>{},
                            mfem::future::Gradient<COORDS>{}, mfem::future::Weight{}},
        {{DISP, &disp->space()}, {VELO, &velo->space()}, {COORDS, mfes}}, std::index_sequence<DISP>{});
    EXPECT_NEAR(mismatch, 0.0, 1.0e-7);
  }

  {
    dfem_weak_form_test::SolidUVACoordsQf qf{.K = mat.K, .G = mat.G};
    auto mismatch = derivativeActionFiniteDifferenceMismatch(
        qf,
        mfem::future::tuple{mfem::future::Gradient<DISP>{}, mfem::future::Gradient<VELO>{},
                            mfem::future::Gradient<ACCEL>{}, mfem::future::Gradient<COORDS>{},
                            mfem::future::Weight{}},
        {{DISP, &disp->space()}, {VELO, &velo->space()}, {ACCEL, &accel->space()}, {COORDS, mfes}},
        std::index_sequence<DISP>{});
    EXPECT_NEAR(mismatch, 0.0, 1.0e-7);
  }

  {
    dfem_weak_form_test::SolidUVACoordsDensityQf qf{.K = mat.K, .G = mat.G};
    auto mismatch = derivativeActionFiniteDifferenceMismatch(
        qf,
        mfem::future::tuple{mfem::future::Gradient<DISP>{}, mfem::future::Gradient<VELO>{},
                            mfem::future::Gradient<ACCEL>{}, mfem::future::Gradient<COORDS>{},
                            mfem::future::Weight{}, mfem::future::Value<DENSITY>{}},
        {{DISP, &disp->space()}, {VELO, &velo->space()}, {ACCEL, &accel->space()}, {COORDS, mfes},
         {DENSITY, &density->space()}},
        std::index_sequence<DISP>{});
    EXPECT_NEAR(mismatch, 0.0, 1.0e-7);
  }
}

// DfemSolidTraction.ConstantTractionMatchesSurfaceIntegral
//
// Goal: apply a constant reference-configuration traction `t = (0, -0.5)` over all
// boundary attributes of a 1.0 x 0.75 Cartesian2D mesh and verify the residual
// satisfies:
//   sum over y-component dofs == -t_y * perimeter = -(-0.5) * 3.5 = 1.75
//   sum over x-component dofs == 0
//
// Currently DISABLED: the residual sums come out asymmetric (x-component is
// nonzero, y-magnitude is ~0.30 of expected) and large residual values land at
// interior nodes (e.g. node (0.25, 0.25)), which the boundary integrator should
// never touch.
//
// Root cause appears to be upstream in mfem::future: when a DifferentiableOperator
// has multiple true input fields (here DISPLACEMENT, VELOCITY, ACCELERATION,
// COORDINATES) and a vector-valued output FES, AddBoundaryIntegrator's restriction
// does not match the residual prolongation done by the Element-based Mult path.
// The single-input boundary test in mfem/tests/unit/dfem/test_mass.cpp works
// because it uses `dop.SetParameters({nodes})` to move coords out of the
// true-input list and operates on a scalar FES.
//
// The smith-side wiring (setTraction, current_time_, qfunction) is in place and
// ready to use once one of these resolves it:
//   (a) MFEM dFEM fixes Entity::BoundaryElement assembly for multi-input
//       vector-FES operators, or
//   (b) DfemWeakForm routes boundary integrals to a separate
//       DifferentiableOperator with minimal infds (coords only).
// See dfem_plan.md for the follow-up.
TEST(DfemSolidTraction, DISABLED_ConstantTractionMatchesSurfaceIntegral)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim = 2;
  constexpr int p = 1;
  constexpr double Lx = 1.0;
  constexpr double Ly = 0.75;
  constexpr int nx = 4;
  constexpr int ny = 3;
  constexpr double traction_y = -0.5;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "dfem_traction_test");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, Lx, Ly), "trac_mesh", 0, 0);

  using VectorSpace = smith::H1<p, dim>;
  smith::FiniteElementState disp(smith::StateManager::newState(VectorSpace{}, "disp", mesh->tag()));
  smith::FiniteElementState velo(smith::StateManager::newState(VectorSpace{}, "velo", mesh->tag()));
  smith::FiniteElementState accel(smith::StateManager::newState(VectorSpace{}, "accel", mesh->tag()));
  smith::FiniteElementState coords(
      *static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(), "coords");
  coords.setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
  disp = 0.0;
  velo = 0.0;
  accel = 0.0;

  auto shape_disp = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());

  auto weak_form = std::make_shared<smith::DfemSolidWeakForm>("solid_trac", mesh, disp.space());

  mfem::Array<int> all_bdr_marker(mesh->mfemParMesh().bdr_attributes.Max());
  all_bdr_marker = 1;

  const mfem::IntegrationRule& bdr_ir =
      mfem::IntRules.Get(mesh->mfemParMesh().GetTypicalFaceGeometry(), 2 * p + 1);

  weak_form->setTraction<dim>(
      all_bdr_marker,
      [](double) {
        mfem::future::tensor<double, dim> trac{};
        trac[1] = traction_y;
        return trac;
      },
      bdr_ir);

  smith::TimeInfo time_info{0.0, 1.0};
  std::vector<smith::ConstFieldPtr> fields{&disp, &velo, &accel, &coords};
  auto residual = weak_form->residual(time_info, shape_disp.get(), fields);

  // Sum of y-component dofs should equal -t_y * perimeter via partition-of-unity.
  const auto& fes = disp.space();
  double sum_x = 0.0, sum_y = 0.0;
  for (int i = 0; i < fes.GetTrueVSize(); ++i) {
    const int comp = (fes.GetOrdering() == mfem::Ordering::byNODES)
                         ? (i / (fes.GetTrueVSize() / dim))
                         : (i % dim);
    if (comp == 0) sum_x += residual[i];
    if (comp == 1) sum_y += residual[i];
  }
  double sum_x_global = 0.0, sum_y_global = 0.0;
  MPI_Allreduce(&sum_x, &sum_x_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_y, &sum_y_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double perimeter = 2.0 * (Lx + Ly);
  EXPECT_NEAR(sum_x_global, 0.0, 1.0e-12);
  EXPECT_NEAR(sum_y_global, -traction_y * perimeter, 1.0e-12);

  smith::StateManager::reset();
}

}  // namespace

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
