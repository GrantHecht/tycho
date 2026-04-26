// Configure-time smoke test for Enzyme Forward-over-Reverse nesting.
// Differentiates grad_cube(x) = d(x^3)/dx via forward mode at x=2,
// asserting the result is f''(2) = 6*2 = 12.
// Exit code 0 = pass, anything else = fail.

extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
}
template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);
template <typename RT, typename... Args> RT __enzyme_autodiff(Args...);

static double cube(double x) { return x * x * x; }   // f''(x) = 6x.

// First derivative of cube via reverse mode (scalar in / scalar out form;
// Enzyme returns the gradient w.r.t. the active by-value argument).
static double grad_cube(double x) {
    return __enzyme_autodiff<double>(reinterpret_cast<void*>(&cube), x);
}

int main() {
    double x = 2.0, dx = 1.0;
    // Forward derivative of grad_cube w.r.t. x at x=2 with tangent 1 -> 6*2 = 12.
    double d2 = __enzyme_fwddiff<double>(reinterpret_cast<void*>(&grad_cube),
                                         x, dx);
    return (d2 > 11.99 && d2 < 12.01) ? 0 : 1;
}
