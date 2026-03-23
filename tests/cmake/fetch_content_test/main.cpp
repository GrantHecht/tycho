#include <tycho/typedefs.h>
#include <iostream>

int main() {
    // Verify core Eigen types are accessible through tycho
    Tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;

    std::cout << "Tycho FetchContent test: OK" << std::endl;
    return 0;
}
