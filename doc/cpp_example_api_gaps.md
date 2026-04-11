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
| ~~MinTimeToClimb~~ | ~~`InterpTable` in ODE~~ | ~~High~~ | ~~RESOLVED (Subsystem 4): `interp()` / `interp_scalar()` free functions + `InterpType` enum. Climb time 319.47 s matches Python ~321 s.~~ | ~~Done~~ |
| MultiPhaseCannon | `add_direct_link_equal_con` on OCP wrapper | Medium | Not on OptimalControlProblem wrapper. Must use `ocp.base().add_direct_link_equal_con(...)` to link ODE parameters between phases. | Forward to OCP wrapper |
| MultiPhaseCannon | ODEParams region + named vars | Medium | `add_lower_var_bound(ODEParams, "rad", ...)` resolves name to full phase vector index instead of raw ODE param index. Must use index-based overload. | Fix name resolution for ODEParams region |
| MultiPhaseCannon | Mixed state+ODE param inequality | Medium | Phase wrapper `add_inequal_con` doesn't support separate ODE param vars. Must use `phase.base().add_inequal_con(region, func, xvars, pvars, spvars, scale)`. | Add overload with param vars |
| DionysusLowThrust | MEE astro models | Medium | Python `MEETwoBody_CSI` and `CSIThruster` not available in C++. MEE dynamics must be implemented inline. | Expose MEE dynamics as C++ builder-compatible ODE factory |
| DionysusLowThrust | `ODE::integrator()` | **High** | Builder `ODE` has no `integrator()` method. Cannot generate initial guesses via integration. Must use manual lerp or external propagation. Blocks faithful porting of ~8 examples that rely on integrated IG. | Add `ODE::integrator()` returning a builder-compatible integrator |
| BettsLowThrust | Zonal gravity via `RowMatrix`/`inverse` | High | Full J2/J3/J4 zonal gravity requires Cartesian-to-RTN rotation via `RowMatrix().inverse()` composition. Same gap as CartPole. | (same fix as CartPole) |
| BettsLowThrust | MEE-to-Cartesian composition | Medium | Full dynamics chain: MEE → Cartesian (via `MEECartFunc`) → zonal gravity → RTN → MEE rates. Multi-level VF composition works in Python but is complex in C++. | Document composition patterns |
| OrbitContinuation | `ODE::integrator()` | **High** | Same gap as DionysusLowThrust. Cannot generate orbital IG via integration. | (same fix) |
| Heteroclinic | `LGLInterpTable` + `make_periodic()` | High | Required for orbit-matching constraints in heteroclinic connection. Not accessible from builder API. | Expose LGLInterpTable for builder API use |
| Heteroclinic | `InterpFunction` | High | Required for time-dependent boundary conditions (orbit position/velocity matching). | Expose InterpFunction for builder API |
| Heteroclinic | STM integration (`integrate_stm_parallel`) | High | Required for manifold computation via monodromy matrix eigendecomposition. Not in builder API. | Expose STM integration for builder API |
| OptimalDocking | `quat_rotate` | High | Python `vf.quat_rotate(q, v)` not available in C++ VF DSL. Only `quat_product` exists. | Add `quat_rotate()` free function to C++ VF DSL |
| OptimalDocking | `cwise_product(Eigen::Vector)` | Medium | `cwise_product()` requires a VF expression argument, not an `Eigen::VectorXd`. Must wrap in `Constant<>` first. | Add overload accepting Eigen vectors |
| OptimalDocking | `InterpTable` for target trajectory | High | Form2 requires interpolated target trajectory for time-dependent BC. Same gap as MinTimeToClimb. | (same fix) |
| MultiSpacecraftOpt | `PhasePack` link constraints | Low | Works but verbose. Must construct `PhasePack` tuples manually and use `ocp.base()`. | Add named-variable link constraint helpers to OCP wrapper |
| MultiSpacecraftOpt | `sub_variables` on Phase wrapper | Medium | Not on Phase wrapper. Must use `phase.base().sub_variables(flag, idx, val)`. | Forward to Phase wrapper |
| ~~MinTimeClimbTables~~ | ~~`InterpTable1D`/`InterpTable2D` in ODE~~ | ~~High~~ | ~~RESOLVED (Subsystem 4): Same fix as MinTimeToClimb. Full table-based aero with `interp_scalar()` and `interp()`.~~ | ~~Done~~ |
| GoddardRocket | Control-law integration for IG | **High** | Python `ode.integrator(dt, Ulaw(), [vars])` propagates ODE with a feedback control law to build initial guesses. No C++ builder equivalent. Blocks faithful IG construction for GoddardRocket, TopputtoLowThrust, MultiPhaseCannon, and others. | Add control-law overload to `ODE::integrator()` |
| MultiPhaseCannon | Event detection in integrator | **High** | Python `integ.integrate_dense(X0, tf, [(event_func, 0, 1)])` integrates until an event triggers (e.g., apogee, ground impact) and splits trajectory at that point. No C++ builder equivalent. Blocks proper phase-split IG construction. | Add event detection to builder integrator |
