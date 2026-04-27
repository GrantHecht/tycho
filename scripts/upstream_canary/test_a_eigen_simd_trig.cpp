// Canary A: can Enzyme differentiate Eigen's vectorised packet trig?
//
// Today (Enzyme tip-of-tree, Eigen 5.0.1): NO.  `Eigen::Array<double, 4, 1>
// ::cos()` lowers to `pcos<Packet4d>` whose IR uses `llvm.x86.avx.round.pd.256`
// + packed `<4 x i64>` xor/and on sign-bit splats, none of which Enzyme can
// handle (upstream issues #689, #2679, #2794).
//
// If this TU starts compiling cleanly, Enzyme has gained support for Eigen's
// SIMD trig path.  At that point we can remove the per-lane scalar libm
// wrappers from `include/tycho/detail/utils/simd_math.h` and call
// `x.cos() / x.sin()` directly inside `tycho::math::cos / sin`.
//
// Build:
//   clang++ -std=c++20 -O3 -march=native -ffast-math -fno-finite-math-only \
//           -fplugin=$ENZYME_PLUGIN -isystem $TYCHO/dep/eigen \
//           test_a_eigen_simd_trig.cpp -o test_a
//
// Pass criterion: TU compiles AND ./test_a prints lane-correct gradients.

#include <Eigen/Core>
#include <cmath>
#include <cstdio>

extern "C" {
extern int enzyme_dup;
extern int enzyme_dupv;
extern int enzyme_const;
extern int enzyme_width;
}

template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);

using V4d = Eigen::Array<double, 4, 1>;

// Use Eigen's SIMD packet trig directly — what Enzyme currently chokes on.
__attribute__((noinline)) void
eigen_cos_W4(const double* x, double* y) {
    Eigen::Map<const V4d> xm(x);
    Eigen::Map<V4d> ym(y);
    ym = xm.cos();   // lowers to pcos<Packet4d>
}

void target(const double* x, double* y) {
    eigen_cos_W4(x, y);
}

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
        std::printf("PASS — Eigen SIMD pcos<Packet4d> differentiates correctly under Enzyme\n");
        return 0;
    }
    return 1;
}
