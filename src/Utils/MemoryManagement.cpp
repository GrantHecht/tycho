#include "MemoryManagement.h"

namespace Tycho {

thread_local MemoryManager::SuperScalarStackType MemoryManager::SuperScalarStack =
    detail::TempStack<DefaultSuperScalar>(TYCHO_DEFAULT_ARENA_SIZE);
thread_local MemoryManager::ScalarStackType MemoryManager::ScalarStack =
    detail::TempStack<double>(TYCHO_DEFAULT_ARENA_SIZE);
bool MemoryManager::UseArena = true;

#ifdef TYCHO_PYTHON_BINDINGS
void MemoryManager::Build(py::module &m) {
    auto obj = py::class_<MemoryManager>(m, "MemoryManager");
    obj.def_static("enable_arena_memory", []() { MemoryManager::enable_arena_memory(); });
    obj.def_static("disable_arena_memory", []() { MemoryManager::disable_arena_memory(); });
}
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho