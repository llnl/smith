#include <vector>
#include <functional>

#include "serac/numerics/functional/tensor.hpp"

struct BC {

    BC(std::function< double(vec3, t) > f, Domain domain, mfem::FiniteElementSpace & fes) {

    }

    void evaluate(mfem::Vector & v, double t) {

    }

    mfem::Array< int > dof_ids;

    bool is_vector_valued;
    std::unique_ptr< mfem::Coefficient > func;
    std::unique_ptr< mfem::VectorCoefficient > vfunc;

};
