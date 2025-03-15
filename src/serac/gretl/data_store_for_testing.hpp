#include "data_store.hpp"

namespace gretl {

struct DataStoreForTesting : public DataStore {
  DataStoreForTesting(size_t maxStates) : DataStore(maxStates) {}

  // reverse back a single state, updating the duals along the way
  StateBase reverse_state(size_t s) override { return DataStore::reverse_state(s); };
};

}  // namespace gretl