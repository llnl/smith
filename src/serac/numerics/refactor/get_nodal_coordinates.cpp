#if 0
#include "serac/numerics/refactor/finite_element.hpp"

namespace refactor {

nd::array< double, 2 > get_nodal_coordinates(const Field & f) {

    uint32_t num_nodes = get_num_nodes(f); 
    uint32_t spatial_dimension = static_cast<uint32_t>(f.mesh().SpaceDimension()); 

    mfem::ParGridFunction pgf = f.gridFunction();

    nd::array<double, 2> nodal_coords({num_nodes, spatial_dimension});

    if (refactor::is_scalar_valued(get_family(f))) {

        for (uint32_t j = 0; j < spatial_dimension; j++) {

            // it seems there is no direct way to get the nodal coordinates
            // for an mfem::GridFunction, so instead we proceed by asking pgf 
            // to evaluate the identity function at each of its nodes 
            mfem::FunctionCoefficient identity_function(
                static_cast<int>(spatial_dimension), 
                [&](const mfem::Vector& x) {
                    output = x[j];
                }
            );

            pgf.ProjectCoefficient(identity_function);

            // then, we extract the nodal coordinates from the gridfunction
            for (uint32_t i = 0; i < num_nodes; i++) {
                //nodal_coords(i, j) = pgf[static_cast<int>(i * spatial_dimension + j)];
                nodal_coords(i, j) = pgf[static_cast<int>(j * num_nodes + i)];
            }

        }

        return nodal_coords;
    
    } else {

        // with vector-valued fields, the workaround is weirder:
        // if we evaluate the function f(x, y, z) = {x, 0, 0}, that will
        // reveal the x-coordinate of the node times the length of the nodal direction
        //
        // so, to get just the coordinates, we need to compute the directions 
        // as well, so we can later divide through by their magnitudes.

        nd::array<double, 2> nodal_directions = get_nodal_directions(f);

        nd::array<double> magnitudes({num_nodes});
        for (uint32_t i = 0; i < num_nodes; i++) {
            double sum = 0.0;
            for (uint32_t d = 0; d < spatial_dimension; d++) {
                sum += nodal_directions(i, d) * nodal_directions(i, d);
            }
            magnitudes[i] = sqrt(sum);
        }

        nd::array<double, 2> nodal_coords({num_nodes, spatial_dimension});

        for (uint32_t d = 0; d < spatial_dimension; d++) {

            mfem::VectorFunctionCoefficient directional_identity_function(
                static_cast<int>(spatial_dimension), 
                [&](const mfem::Vector& x, mfem::Vector & output) {
                    output = 0.0;
                    output[static_cast<int>(d)] = x[static_cast<int>(d)];
                }
            );

            pgf.ProjectCoefficient(directional_identity_function);

            for (uint32_t i = 0; i < num_nodes; i++) {
                nodal_coords(i, d) = pgf[static_cast<int>(i * spatial_dimension + d)];
            }

        }

        return nodal_coords;
    

    }
    
}

}
#endif
