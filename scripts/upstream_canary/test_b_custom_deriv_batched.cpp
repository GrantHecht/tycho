// Canary B: do Enzyme custom-registered derivatives compose with
// `enzyme_width > 1` (Phase 3 batched tangents)?
//
// Today (Enzyme tip-of-tree): NO.  Enzyme generates an internal
// `fixderivative_*` bridge whose signature uses scalar shadow types
// regardless of width, but the call site passes `[width x ptr]`
// aggregates -> "Call parameter type does not match function signature"
// IR verification crash.
//
// If this TU starts compiling cleanly, Enzyme has fixed the
// custom-derivative + batching composition.  At that point we have a
// second viable architecture: keep Eigen's SIMD packet trig in the
// wrapper body, register a custom derivative, and let Phase 3 batching
// amortise across BW tangents — potentially faster than the current
// scalar-libm-trig approach.
//
// Build:
//   clang++ -std=c++20 -O3 -fplugin=$ENZYME_PLUGIN \
//           test_b_custom_deriv_batched.cpp -o test_b
//
// Pass criterion: TU compiles AND ./test_b prints lane-correct gradients.

#include <cmath>
#include <cstdio>

extern "C" {
extern int enzyme_dup;
extern int enzyme_dupv;
extern int enzyme_const;
extern int enzyme_width;
}

template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);

extern "C" {

__attribute__((noinline)) void
my_cos_W4(const double* x, double* y) {
    for (int k = 0; k < 4; ++k) y[k] = std::cos(x[k]);
}

__attribute__((noinline)) void
my_cos_W4_fwd(const double* x, const double* dx,
              double* y, double* dy) {
    for (int k = 0; k < 4; ++k) {
        y[k] = std::cos(x[k]);
        dy[k] = -std::sin(x[k]) * dx[k];
    }
}

void* __enzyme_register_derivative_my_cos_W4[] = {
    reinterpret_cast<void*>(&my_cos_W4),
    reinterpret_cast<void*>(&my_cos_W4_fwd),
};

void target(const double* x, double* y) {
    my_cos_W4(x, y);
}

} // extern "C"

constexpr int BW = 4;

int main() {
    alignas(32) double x[4]  = {0.5, 1.0, 1.5, 2.0};
    alignas(32) double y[4]  = {0};
    alignas(32) double dx[BW * 4] = {0};
    alignas(32) double dy[BW * 4] = {0};
    for (int k = 0; k < BW; ++k) dx[k * 4 + k] = 1.0;

    const long stride_x = 4 * sizeof(double);
    const long stride_y = 4 * sizeof(double);

    __enzyme_fwddiff<void>(
        reinterpret_cast<void*>(&target),
        enzyme_width, BW,
        enzyme_dupv, stride_x, x, dx,
        enzyme_dupv, stride_y, y, dy);

    int errors = 0;
    for (int k = 0; k < BW; ++k) {
        double expected = -std::sin(x[k]);
        double got = dy[k * 4 + k];
        if (std::abs(got - expected) > 1e-12) {
            std::fprintf(stderr, "lane %d: expected %.15g, got %.15g\n",
                         k, expected, got);
            ++errors;
        }
    }
    if (errors == 0) {
        std::printf("PASS — custom derivative composes with enzyme_width=%d\n", BW);
        return 0;
    }
    return 1;
}
