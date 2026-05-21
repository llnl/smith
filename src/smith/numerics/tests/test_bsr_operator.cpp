#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/bsr_operator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace {

std::unique_ptr<mfem::HypreParMatrix> makeElasticityMatrix(int dim)
{
  std::unique_ptr<mfem::Mesh> mesh;
  if (dim == 2) {
    mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian2D(6, 5, mfem::Element::QUADRILATERAL));
  } else {
    mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian3D(4, 3, 3, mfem::Element::HEXAHEDRON));
  }

  mfem::ParMesh pmesh(MPI_COMM_WORLD, *mesh);
  mfem::H1_FECollection fec(1, dim);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, dim, mfem::Ordering::byVDIM);

  mfem::ConstantCoefficient lambda(1.0), mu(1.0);
  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
  a.Assemble();
  a.Finalize();
  return std::unique_ptr<mfem::HypreParMatrix>(a.ParallelAssemble());
}

double globalMaxError(const mfem::Vector& a, const mfem::Vector& b)
{
  double local = 0.0;
  for (int i = 0; i < a.Size(); ++i) {
    local = std::max(local, std::abs(a(i) - b(i)));
  }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

void fillInput(mfem::Vector& x)
{
  for (int i = 0; i < x.Size(); ++i) {
    x(i) = std::sin(0.17 * (i + 1)) + 0.25 * std::cos(0.11 * (i + 3));
  }
}

void expectBSRMatchesHypre(int dim)
{
  auto A = makeElasticityMatrix(dim);
  smith::BSROperator bsr(A.get(), dim);
  EXPECT_TRUE(bsr.Enabled());

  mfem::Vector x(A->Width()), y_hypre(A->Height()), y_bsr(A->Height());
  fillInput(x);

  A->Mult(x, y_hypre);
  bsr.Mult(x, y_bsr);
  EXPECT_LT(globalMaxError(y_hypre, y_bsr), 1.0e-12);

  y_hypre = 1.0;
  y_bsr = 1.0;
  A->AddMult(x, y_hypre, -0.5);
  bsr.AddMult(x, y_bsr, -0.5);
  EXPECT_LT(globalMaxError(y_hypre, y_bsr), 1.0e-12);
}

}  // namespace

TEST(BSROperator, Elasticity2DMatchesHypre)
{
  expectBSRMatchesHypre(2);
}

TEST(BSROperator, Elasticity3DMatchesHypre)
{
  expectBSRMatchesHypre(3);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
