#include "MemoryManagerBind.h"
#include "MemoryManagement.h"

using namespace Tycho;

void TychoBind<MemoryManager>::Build(nb::module_ &m) {
    auto obj = nb::class_<MemoryManager>(m, "MemoryManager");
    obj.def_static("enable_arena_memory", []() { MemoryManager::enable_arena_memory(); });
    obj.def_static("disable_arena_memory", []() { MemoryManager::disable_arena_memory(); });
}
