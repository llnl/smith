#pragma once

#include "serac/mesh/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#define SERAC_MESH_DIR SERAC_REPO_DIR "/data/meshes/"

mfem::ParMesh load_parmesh(std::string filename) {
    mfem::Mesh mesh(filename);
    return mfem::ParMesh(*serac::mesh::refineAndDistribute(std::move(mesh), 0, 0));
}
