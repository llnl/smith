// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/about.hpp"

#include "serac/serac_config.hpp"

#include "axom/config.hpp"
#include "axom/core.hpp"
#include "axom/fmt.hpp"

#include "camp/config.hpp"

#ifdef SERAC_USE_CALIPER
#include "caliper/caliper-config.h"
#endif

#ifdef SERAC_USE_CONDUIT
#include "conduit_config.h"
#endif

#ifdef SERAC_USE_HDF5
#include "hdf5.h"
#endif

#ifdef SERAC_USE_LUA
#include "lua.h"
#endif

#include "mfem.hpp"

#ifdef SERAC_USE_RAJA
#include "RAJA/config.hpp"
#endif

#ifdef SERAC_USE_UMPIRE
#include "umpire/Umpire.hpp"
#endif

#ifdef SERAC_USE_TRIBOL
#include "tribol/config.hpp"
#endif

#include "serac/infrastructure/git_sha.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"

namespace serac {

std::string about()
{
  using namespace axom::fmt;
  [[maybe_unused]] constexpr std::string_view on = "ON";
  [[maybe_unused]] constexpr std::string_view off = "OFF";

  std::string about = "\n";

  // Version info
  about += format("Serac Version:   {0}\n", version(false));
  about += format("Git Commit SHA:  {0}\n", gitSHA());
  about += "\n";

  // General configuration
#ifdef SERAC_DEBUG
  about += format("Debug Build:     {0}\n", on);
#else
  about += format("Debug Build:     {0}\n", off);
#endif

#ifdef SERAC_USE_CUDA
  about += format("CUDA:            {0}\n", on);
#else
  about += format("CUDA:            {0}\n", off);
#endif

  about += "\n";

  //------------------------
  // Libraries
  //------------------------

  // Print out version of enabled libraries and list disabled ones by name

  std::vector<std::string> disabled_libs;

  about += "Enabled Libraries:\n";

  // Axom
  about += format("Axom Version:    {0}\n", axom::getVersion());

  // Camp
  about += format("Camp Version:    {0}\n", CAMP_VERSION);

  // Caliper
#ifdef SERAC_USE_CALIPER
  about += format("Caliper Version: {0}\n", CALIPER_VERSION);
#else
  disabled_libs.push_back("Caliper");
#endif

  // Conduit
#ifdef SERAC_USE_CONDUIT
  about += format("Conduit Version: {0}\n", CONDUIT_VERSION);
#else
  disabled_libs.push_back("Conduit");
#endif

  // HDF5
#ifdef SERAC_USE_HDF5
  unsigned int h5_maj, h5_min, h5_rel;
  std::string h5_version;
  if (H5get_libversion(&h5_maj, &h5_min, &h5_rel) < 0) {
    SLIC_ERROR("Failed to retrieve HDF5 version.");
  } else {
    h5_version = format("{0}.{1}.{2}", h5_maj, h5_min, h5_rel);
  }
  about += format("HDF5 Version:    {0}\n", h5_version);
#else
  disabled_libs.push_back("HDF5");
#endif

  // Lua
#ifdef SERAC_USE_LUA
  std::string lua_version{LUA_RELEASE};
  if (axom::utilities::string::startsWith(lua_version, "Lua ")) {
    lua_version.erase(0, 4);
  }
  about += format("Lua Version:     {0}\n", lua_version);
#else
  disabled_libs.push_back("Lua");
#endif

  // MFEM
  const char* mfem_version = mfem::GetVersionStr();
  if (mfem_version == nullptr) {
    SLIC_ERROR("Failed to retrieve MFEM version.");
  }
  const char* mfem_sha = mfem::GetGitStr();
  if (mfem_sha == nullptr) {
    SLIC_ERROR("Failed to retrieve MFEM Git SHA.");
  }
  std::string mfem_full_version = std::string(mfem_version);
  if (axom::utilities::string::startsWith(mfem_full_version, "MFEM ")) {
    mfem_full_version.erase(0, 5);
  }
  if (mfem_sha[0] != '\0') {
    mfem_full_version += format(" (Git SHA: {0})", mfem_sha);
  }
  about += format("MFEM Version:    {0}\n", mfem_full_version);

  // RAJA
#ifdef SERAC_USE_RAJA
  about += format("RAJA Version:    {0}.{1}.{2}\n", RAJA_VERSION_MAJOR, RAJA_VERSION_MINOR, RAJA_VERSION_PATCHLEVEL);
#else
  disabled_libs.push_back("RAJA");
#endif

  // Tribol
#ifdef SERAC_USE_TRIBOL
  about += format("Tribol Version:    {0}\n", TRIBOL_VERSION_FULL);
#else
  disabled_libs.push_back("Tribol");
#endif

  // Umpire
#ifdef SERAC_USE_UMPIRE
  about += format("Umpire Version:  {0}.{1}.{2}\n", umpire::get_major_version(), umpire::get_minor_version(),
                  umpire::get_patch_version());
#else
  disabled_libs.push_back("Umpire");
#endif

  about += "\n";

  about += "Disabled Libraries:\n";
  if (disabled_libs.size() == 0) {
    about += "None\n";
  } else {
    for (auto& lib : disabled_libs) {
      about += lib + "\n";
    }
  }

  return about;
}

std::string gitSHA() { return SERAC_GIT_SHA; }

void printRunInfo()
{
  // Add header
  std::string infoMsg = axom::fmt::format("\n{:*^80}\n", "Run Information");

  infoMsg += axom::fmt::format("{0}: {1}\n", "Version", version());
  infoMsg += axom::fmt::format("{0}: {1}\n", "User Name", axom::utilities::getUserName());
  infoMsg += axom::fmt::format("{0}: {1}\n", "Host Name", axom::utilities::getHostName());

  auto [count, rank] = getMPIInfo();
  infoMsg += axom::fmt::format("{0}: {1}\n", "MPI Rank Count", count);

  // Add footer
  infoMsg += axom::fmt::format("{:*^80}\n", "*");

  SLIC_INFO_ROOT(infoMsg);
  serac::logger::flush();
}

std::string version(bool add_SHA)
{
  std::string version =
      axom::fmt::format("v{0}.{1}.{2}", SERAC_VERSION_MAJOR, SERAC_VERSION_MINOR, SERAC_VERSION_PATCH);

  std::string sha = gitSHA();
  if (add_SHA && !sha.empty()) {
    version += "-" + sha;
  }

  return version;
}

std::string compiler() { return axom::fmt::format("{0} version {1}", SERAC_COMPILER_NAME, SERAC_COMPILER_VERSION); }

}  // namespace serac
