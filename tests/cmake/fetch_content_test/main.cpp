// Verify core Eigen types and fmt dependency are accessible through tycho
#include <tycho/typedefs.h>
#include <tycho/utils.h>
#include <iostream>

int main() {
    Tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho FetchContent test: OK (norm=" << v.norm() << ")"
              << std::endl;
    return 0;
}
