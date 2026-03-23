#include <tycho/typedefs.h>
#include <iostream>

int main() {
    Tycho::Vector3<double> v;
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho find_package test: OK" << std::endl;
    return 0;
}
