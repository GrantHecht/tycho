# First-class variable bounds in PSIOPT — implemented

A phase's `add_lu_var_bound` / `add_lower_var_bound` / `add_upper_var_bound`
declarations are recorded as `(region, variable, lower, upper)` and resolved at
transcription into bounds on the NLP's decision variables (`NonLinearProgram`'s
`x_lower_` / `x_upper_`), instead of being lowered into inequality constraints.
PSIOPT consumes that contract natively: the log-barrier is applied to the bounded
variables themselves with bound multipliers `z_L` / `z_U` eliminated into the
(1,1) block's `Sigma` diagonal, so a bound costs no constraint row, no slack, and
no inequality multiplier. The declarations therefore take no constraint scale —
conditioning of a bounded variable is a job for variable scaling — and repeated
declarations on the same variable intersect, with an empty intersection raised at
transcription.

See `ODEPhaseBase::record_var_bounds` / `transcribe_var_bounds` in
`src/optimal_control/ode_phase_base.cpp` for the phase side, and the bound set,
barrier, and fraction-to-boundary handling in
`include/tycho/detail/solvers/psiopt.h` for the solver side.
