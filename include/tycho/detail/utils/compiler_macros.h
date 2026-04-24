#pragma once

// Portable always-inline attribute. Use on short, frequently-inlined hot-path
// wrappers where the compiler's default inlining heuristic can miss.
// GCC/Clang: [[gnu::always_inline]] (C++11 attribute, clang-cl supports it).
// MSVC:     __forceinline (pre-C++11 keyword; ignores the attribute syntax).
// Other:    plain inline fallback.
#if defined(__GNUC__) || defined(__clang__)
#define TYCHO_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define TYCHO_ALWAYS_INLINE __forceinline
#else
#define TYCHO_ALWAYS_INLINE inline
#endif
