// Configure-time smoke test for Enzyme Forward-over-Forward nesting.
// Differentiates dcube(x, dx) = f'(x) * dx via forward mode w.r.t. x at x=2
// with inner tangent dx=1 held constant, asserting f''(2) * 1 = 12.
// Exit code 0 = pass, anything else = fail.

extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
}
template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);

static double cube(double x) { return x * x * x; }   // f''(x) = 6x.

// First forward derivative of cube. Active first arg, second is its tangent.
static double dcube(double x, double dx) {
    return __enzyme_fwddiff<double>(reinterpret_cast<void*>(&cube), x, dx);
}

int main() {
    double x = 2.0;
    double dx = 1.0;   // inner tangent (held constant under outer fwd)
    // Outer fwd of dcube w.r.t. its first arg, seeding x with 1.0 and
    // marking dx as constant. Result is f''(x) * dx = 6*2*1 = 12.
    double d2 = __enzyme_fwddiff<double>(
        reinterpret_cast<void*>(&dcube),
        x, 1.0,
        enzyme_const, dx);
    return (d2 > 11.99 && d2 < 12.01) ? 0 : 1;
}
