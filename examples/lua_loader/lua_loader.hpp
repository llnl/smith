#pragma once

#include <mpi.h>
#include <string>
#include <sstream>
#include "mfem.hpp"
#include "serac/serac.hpp"
#include <sol/sol.hpp>
#include <filesystem>

namespace LuaLoader {

/**
 * @brief Class for loading and interacting with Lua scripts using the sol library.
 */
class LuaLoader {
 public:
  /**
   * @brief Constructor. Loads a Lua script, sets up the Lua environment, and executes the script.
   * @param script_path_ Path to the Lua script file.
   * @param verbose_ If true, enables verbose logging.
   */
  LuaLoader(std::filesystem::path script_path_, bool verbose_);

  /// Type alias for a function taking (vec3, double) and returning vec3
  using xtfunc = std::function<serac::vec3(const serac::vec3&, const double)>;

  /**
   * @brief Extracts a single parameter from Lua, with a default value fallback.
   * @tparam T Type of the parameter to extract.
   * @param parameter_name Name of the parameter in the Lua script.
   * @param default_value Value to return if the parameter is not found in Lua.
   * @return The extracted parameter value, or the default value if not found.
   */
  template <typename T>
  T ExtractLuaParameter(std::string parameter_name, T default_value);

  /**
   * @brief Extracts a sub-table from a Lua table, with a default value fallback.
   * @tparam T Type of the elements in the vector to extract.
   * @param parameter_name Name of the table in the Lua script.
   * @param sub_name Name of the sub-table or entry in the Lua table.
   * @param default_value Value to return if the table or sub-table is not found in Lua.
   * @return The extracted vector, or the default value if not found.
   */
  template <typename T>
  std::vector<T> ExtractLuaTable(std::string parameter_name, std::string sub_name, std::vector<T> default_value);

  /**
   * @brief Extracts a space-time function from Lua, wraps it in a C++ std::function, or returns default.
   * @param parameter_name Name of the function in the Lua script.
   * @param default_value Function to return if the Lua function is not found.
   * @return The extracted function as xtfunc, or the default function if not found.
   */
  xtfunc ExtractLuaSpaceTimeFunction(std::string parameter_name, xtfunc default_value);
  std::function<double(double)> ExtractLuaScalarFunction(std::string parameter_name, std::function<double(double)>);

 private:
  sol::state lua;                     ///< The Lua state/context
  std::filesystem::path script_path;  ///< Path to the Lua script
  std::string script_dir;             ///< Directory containing the script
  bool verbose = false;               ///< Verbosity flag for logging
};

// Function definitions with Doxygen comments

inline LuaLoader::LuaLoader(std::filesystem::path script_path_, bool verbose_)
    : script_path(script_path_), verbose(verbose_)
{
  script_dir = script_path.parent_path().string();
  lua.open_libraries(sol::lib::base, sol::lib::package, sol::lib::math, sol::lib::table, sol::lib::string);
  std::string pkg_path = lua["package"]["path"];
  pkg_path += ";" + script_dir + "/?.lua;" + script_dir + "/?/init.lua";
  lua["package"]["path"] = pkg_path;

  sol::load_result chunk = lua.load_file(script_path.string());
  if (!chunk.valid()) {
    sol::error err = chunk;
    std::cerr << "load error:\n" << err.what() << "\n";
    // return 2;
  }

  // Execute in protected mode
  sol::protected_function_result exec_res = chunk();
  if (!exec_res.valid()) {
    sol::error err = exec_res;
    std::cerr << "runtime error:\n" << err.what() << "\n";
    // return 3;
  }
}

template <typename T>
T LuaLoader::ExtractLuaParameter(std::string parameter_name, T default_value)
{
  sol::optional<T> lua_val = lua[parameter_name];
  if (lua_val.has_value()) {
    T val = lua_val.value();
    if (verbose) {
      SLIC_INFO_ROOT(axom::fmt::format("Lua parameter: {}, value: {}", parameter_name, val));
    }
    return val;
  } else {
    SLIC_INFO_ROOT(axom::fmt::format("Lua parameter: {}, value (default): {}", parameter_name, default_value));
    return default_value;
  }
}

template <typename T>
std::vector<T> LuaLoader::ExtractLuaTable(std::string parameter_name, std::string sub_name,
                                          std::vector<T> default_value)
{
  sol::optional<sol::table> lua_val = lua[parameter_name];
  if (lua_val.has_value()) {
    sol::table mytable = lua_val.value();
    std::vector val = mytable[sub_name].get_or(std::vector<T>{});
    if (verbose) {
      std::stringstream ss;
      ss << "Lua parameter, sub_name: (" << parameter_name << "," << sub_name << ", value: ";
      for (size_t i = 0; i < val.size(); i++) {
        ss << val[i] << ",";
      }
      ss << ")\n";

      SLIC_INFO_ROOT(axom::fmt::format(ss.str()));
    }

    return val;
  } else {
    if (verbose) {
      std::stringstream ss;
      ss << "Lua parameter, sub_name: " << parameter_name << "," << sub_name << ", value (default): ";
      for (size_t i = 0; i < default_value.size(); i++) {
        ss << default_value[i] << ",";
      }
      ss << "\n";

      SLIC_INFO_ROOT(axom::fmt::format(ss.str()));
    }
    return default_value;
  }
}
inline std::function<double(double)> LuaLoader::ExtractLuaScalarFunction(
    std::string parameter_name, std::function<double(const double)> default_value)
{
  sol::optional<sol::function> lua_val = lua[parameter_name];
  if (lua_val.has_value()) {
    sol::function lua_func = lua_val.value();
    std::function<double(const double)> myfunc = [lua_func](const double t) -> double {
      sol::protected_function_result r = lua_func(t);
      if (!r.valid()) {
        // Handle error if Lua function call fails
        sol::error err = r;
        throw std::runtime_error("Lua function call failed: " + std::string(err.what()));
      }
      // Extract the double value from the result
      double value = r;
      return value;
    };

    return myfunc;
  } else {
    return default_value;
  }
};

inline LuaLoader::xtfunc LuaLoader::ExtractLuaSpaceTimeFunction(std::string parameter_name, xtfunc default_value)
{
  sol::optional<sol::function> lua_val = lua[parameter_name];
  if (lua_val.has_value()) {
    sol::function lua_func = lua_val.value();
    xtfunc myfunc = [lua_func](const serac::vec3& x, const double t) -> serac::vec3 {
      sol::protected_function_result r = lua_func(x[0], x[1], x[2], t);
      sol::table table = r.get<sol::table>();
      return serac::vec3{table[1], table[2], table[3]};
    };
    SLIC_INFO_ROOT(axom::fmt::format("Loaded Custom Function for {}", parameter_name));
    return myfunc;
  } else {
    return default_value;
  }
}

}  // namespace LuaLoader