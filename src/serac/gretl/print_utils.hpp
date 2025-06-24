#include <string>
#include <fstream>
#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace gretl {

/**
 * @brief Return string of given parameter's type
 * @tparam T the type to get a string name for
 * @param[in] var the variable to get the type of
 * @return string representation of the type
 */
template <typename T>
std::string typeString(T& var)
{
  // Remove reference, but keep the const/volatile qualifiers.
  const char* name = typeid(var).name();
#ifdef __GNUG__
  int status = -4;  // Arbitrary value to eliminate the compiler warning
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string result((status == 0) ? demangled : name);
  std::free(demangled);
  if constexpr (std::is_const_v<T>) {
    result = "const " + result;
  }
  return result;
#else
  // Return name if compiler doesn't support GNU's extensions (most do)
  return name;
#endif
}

// Base case for the recursive variadic macro
inline void print() { std::cout << std::endl; }

// Recursive case for the variadic macro
template <typename T, typename... Args>
void print(T value, Args... args)
{
  // Find the current variable name in the comma-separated string
  std::cout << value << " ";
  print(args...);  // Recurse for remaining arguments
}

// Base case for the recursive variadic macro
inline void printt() { std::cout << std::endl; }

// Recursive case for the variadic macro
template <typename T, typename... Args>
void printt(T value, Args... args)
{
  // Find the current variable name in the comma-separated string
  std::cout << value << " (" << typeString(value) << ") ";
  printt(args...);  // Recurse for remaining arguments
}

// Base case for the recursive variadic macro
inline void printtype() { std::cout << std::endl; }

// Recursive case for the variadic macro
template <typename T, typename... Args>
void printtype(T value, Args... args)
{
  // Find the current variable name in the comma-separated string
  std::cout << typeString(value) << " ";
  printtype(args...);  // Recurse for remaining arguments
}

}  // namespace gretl