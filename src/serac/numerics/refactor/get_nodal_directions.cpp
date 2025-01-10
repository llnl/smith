#if 0
#include "serac/numerics/refactor/finite_element.hpp"

namespace refactor {

nd::array< double, 2 > get_nodal_directions(const Field & f) {

    if (refactor::is_scalar_valued(get_family(f))) {
        SLIC_ERROR("invalid function space for get_nodal_directions");
    }

    uint32_t num_nodes = get_num_nodes(f); 
    uint32_t spatial_dimension = static_cast<uint32_t>(f.mesh().SpaceDimension()); 

    mfem::ParGridFunction pgf = f.gridFunction();

    // it seems there is no direct way to get the nodal directions
    // for a mfem::GridFunction, so instead we proceed by asking pgf 
    // to evaluate the identity function at each of its nodes 
    mfem::VectorFunctionCoefficient identity_function(
        static_cast<int>(spatial_dimension), 
        [&](const mfem::Vector& x, mfem::Vector & output) {
            output = x;
        }
    );

    pgf.ProjectCoefficient(identity_function);

    // then, we extract the nodal coordinates from the gridfunction
    nd::array<double, 2> nodal_coords({num_nodes, spatial_dimension});

    for (uint32_t i = 0; i < num_nodes; i++) {
        for (uint32_t j = 0; j < spatial_dimension; j++) {
            nodal_coords(i, j) = pgf[static_cast<int>(i * spatial_dimension + j)];
        }
    }

    return nodal_coords;
    
}

}
#endif
