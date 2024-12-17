#include "common.hpp"

namespace refactor {

uint32_t elements_per_block(mfem::Geometry::Type geom, Family family, int p) {

  uint32_t gid = uint32_t(geom);
  uint32_t fid = uint32_t(family);

  uint32_t values[6][2][4] = {

    // vertex
    {{{}}},

    // edge
    {
      {0, 32, 16, 16}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // triangle
    {
      {0, 32, 16, 16}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // quadrilateral
    {
      {0, 32, 14, 8}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // tetrahedron
    {
      {0, 12, 8, 8}, // H1
      {0,  1, 1, 1}, // Hcurl
    },

    // hexahedron
    {
      {0, 8, 4, 2}, // H1
      {0, 1, 1, 1}, // Hcurl
    },
  };

  return values[gid][fid][p];

}

} // namespace refactor
