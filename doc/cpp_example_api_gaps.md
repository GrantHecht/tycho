# C++ Builder API Gaps (discovered during example porting)

| Example | Feature | Severity | Description | Suggested Fix |
|---------|---------|----------|-------------|---------------|
| BikeObstacle | `arctan()` naming | Low | Python bindings expose `vf.arctan()` but C++ uses `atan()` | Document in migration guide |
| VanDerPol | `set_tols()` signature | Low | Python `set_tols(a,b,c)` takes 3 args, C++ `set_tols(kkt,econ,icon,bar)` takes 4 | Align signatures or document |
| CartPole | `RowMatrix`/`inverse` | Medium | Python `vf.RowMatrix(vec, rows, cols).inverse() * Q` not available in C++ expression DSL. Must use analytic inverse or low-level `MatrixFunctionView`/`MatrixInverse`/`MatrixFunctionProduct` types. | Add `RowMatrix()`/`.inverse()` free functions or member functions in C++ VF DSL |
| ParallelParking | Mixed var sources in `add_inequal_con` | Medium | Phase wrapper `add_inequal_con` only accepts phase var names/indices, not mixed `(xtup, OP, SP)`. Must use `phase.base().add_inequal_con(flag, func, xtup, op, sp, scale)`. | Add named-variable overload accepting static param names |
| ParallelParking | `refine_traj_manual` | Medium | Not on Phase wrapper. Must use `phase.base().refine_traj_manual(ndef)`. | Forward to Phase wrapper |
| ParallelParking | `sub_variable` | Medium | Not on Phase wrapper. Must use `phase.base().sub_variable(flag, idx, val)`. | Forward to Phase wrapper |
| ParallelParking | `LerpIG` in `ODE::phase()` | Medium | Builder `ODE::phase(mode, traj, nsegs)` has no LerpIG flag for sparse waypoint interpolation. Must construct with dummy IG then `phase.base().set_traj(waypoints, ndef, true)`. | Add 4th `bool lerp_ig` parameter to builder `ODE::phase()` |
| HangingChain | `Jet.map()` parallel solve | Medium | Python uses `Jet.map()` for batch solving over parameter sweeps. No C++ builder equivalent. | Expose Jet parallel solve for Phase wrapper |
| Reentry | `refine_traj_manual` | Medium | Same gap as ParallelParking — not on Phase wrapper | (same fix) |
| MinTimeToClimb | `InterpTable` in ODE | High | Python uses `InterpTable` for aero/atmospheric data lookup inside ODE. C++ builder API has no equivalent. Simplified with polynomial fits for this example. | Expose `InterpTable` for use in builder `define()` lambdas |
| MultiPhaseCannon | `add_direct_link_equal_con` on OCP wrapper | Medium | Not on OptimalControlProblem wrapper. Must use `ocp.base().add_direct_link_equal_con(...)` to link ODE parameters between phases. | Forward to OCP wrapper |
| MultiPhaseCannon | ODEParams region + named vars | Medium | `add_lower_var_bound(ODEParams, "rad", ...)` resolves name to full phase vector index instead of raw ODE param index. Must use index-based overload. | Fix name resolution for ODEParams region |
| MultiPhaseCannon | Mixed state+ODE param inequality | Medium | Phase wrapper `add_inequal_con` doesn't support separate ODE param vars. Must use `phase.base().add_inequal_con(region, func, xvars, pvars, spvars, scale)`. | Add overload with param vars |
