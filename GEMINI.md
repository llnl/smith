# Smith Project — Gemini Code Guidelines

## Environment & Build
- The main build directory is `build`.
- Run tests quickly using `cd build && ctest -j 8`. To run a specific test, use `ctest -R "[testname]"` (e.g., `ctest -R "test_thermo_mechanics"`).
- Always compile the project using `make -j 8 -C build` before running tests.
- Maintain existing codebase conventions when writing tests (e.g. `SMITH_MARK_FUNCTION`).

## C++ Style Conventions

### Private Member Variables
Private member variables of classes must end with a trailing underscore (`_`).

```cpp
// Good
class Foo {
 private:
  int count_;
  std::string name_;
};

// Bad
class Foo {
 private:
  int count;
  std::string name;
};
```

This convention does **not** apply to:
- Public or protected data members of `struct` types used as plain data carriers
- Local variables or function parameters
- Private *methods* (only data members)

## Git Protocol
- If requested to commit, craft descriptive and concise commit messages.
- Always add new files explicitly using `git add <file>` before committing.
- Do NOT perform a `git push` unless expressly instructed to by the user.