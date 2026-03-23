// Verify core Eigen types and compile definitions are accessible via find_package(tycho)
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

int main() {
    Tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho find_package test: OK (norm=" << v.norm() << ")"
              << std::endl;
    return 0;
}
