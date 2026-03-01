#include "MemoryManagement.h"

namespace Tycho {

thread_local MemoryManager::SuperScalarStackType MemoryManager::SuperScalarStack =
    detail::TempStack<DefaultSuperScalar>(TYCHO_DEFAULT_ARENA_SIZE);
thread_local MemoryManager::ScalarStackType MemoryManager::ScalarStack =
    detail::TempStack<double>(TYCHO_DEFAULT_ARENA_SIZE);
bool MemoryManager::UseArena = true;

} // namespace Tycho
