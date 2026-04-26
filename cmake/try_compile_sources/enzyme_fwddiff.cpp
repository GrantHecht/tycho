// Configure-time smoke test for Enzyme forward-mode AD.
// Differentiates f(x) = x*x at x=3, asserts f'(3) == 6.
// Exit code 0 = pass, anything else = fail.

extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
}
template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);

static double square(double x) { return x * x; }

int main() {
    double x = 3.0, dx = 1.0;
    double dy = __enzyme_fwddiff<double>(reinterpret_cast<void*>(&square),
                                         enzyme_dup, x, dx);
    // Allow tiny numerical noise even though this should be exact.
    return (dy > 5.999 && dy < 6.001) ? 0 : 1;
}
