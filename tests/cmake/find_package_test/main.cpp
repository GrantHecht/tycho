// Verify core Eigen types and fmt dependency are accessible via find_package(tycho)
#include <tycho/typedefs.h>
#include <tycho/utils.h>
#include <iostream>

int main() {
    Tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho find_package test: OK (norm=" << v.norm() << ")"
              << std::endl;
    return 0;
}
