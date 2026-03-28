// Verify that linking tycho::tycho propagates required include paths and
// compile definitions. Uses typedefs.h (Eigen) and utils.h (fmt via flat_map.h)
// rather than the umbrella <tycho/tycho.h> because some public detail headers
// use fmt/color.h APIs that trigger GCC 15 -Wtemplate-body errors. The umbrella
// compiles correctly with the project's Clang toolchain.
#include <tycho/typedefs.h>
#include <tycho/utils.h>
#include <iostream>

#ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#error "EIGEN_INITIALIZE_MATRICES_BY_ZERO must be defined by tycho"
#endif
#ifndef EIGEN_DONT_PARALLELIZE
#error "EIGEN_DONT_PARALLELIZE must be defined by tycho"
#endif
#ifndef FMT_HEADER_ONLY
#error "FMT_HEADER_ONLY must be defined by tycho"
#endif
#if FMT_USE_LOCALE != 0
#error "FMT_USE_LOCALE must be 0 (set by tycho)"
#endif
#ifndef TYCHO_DEFAULT_QP_THREADS
#error "TYCHO_DEFAULT_QP_THREADS must be defined by tycho"
#endif

#ifndef TYCHO_TEST_NAME
#define TYCHO_TEST_NAME "unknown"
#endif

int main() {
    tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho " TYCHO_TEST_NAME " test: OK (norm=" << v.norm() << ")"
              << std::endl;
    return 0;
}
