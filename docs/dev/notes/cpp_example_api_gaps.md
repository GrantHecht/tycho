# C++ Builder API Gaps (discovered during example porting)

| Example | Feature | Severity | Description | Suggested Fix |
|---------|---------|----------|-------------|---------------|
| BikeObstacle | `arctan()` naming | Low | Python bindings expose `vf.arctan()` but C++ uses `atan()` | Document in migration guide |
| VanDerPol | `set_tols()` signature | Low | Python `set_tols(a,b,c)` takes 3 args, C++ `set_tols(kkt,econ,icon,bar)` takes 4 | Align signatures or document |
| ~~CartPole~~ | ~~`RowMatrix`/`inverse`~~ | ~~Medium~~ | ~~RESOLVED: `row_matrix()` free function + `.inverse()` member in VF DSL~~ | ~~Done~~ |
| ~~ParallelParking~~ | ~~Mixed var sources in `add_inequal_con`~~ | ~~Medium~~ | ~~RESOLVED: Phase wrapper mixed-variable overloads~~ | ~~Done~~ |
| ~~ParallelParking~~ | ~~`refine_traj_manual`~~ | ~~Medium~~ | ~~RESOLVED: Forwarded to Phase wrapper~~ | ~~Done~~ |
| ~~ParallelParking~~ | ~~`sub_variable`~~ | ~~Medium~~ | ~~RESOLVED: Forwarded to Phase wrapper~~ | ~~Done~~ |
| ~~ParallelParking~~ | ~~`LerpIG` in `ODE::phase()`~~ | ~~Medium~~ | ~~RESOLVED: `bool lerp_ig` parameter on `ODE::phase()`~~ | ~~Done~~ |
| ~~HangingChain~~ | ~~`Jet.map()` parallel solve~~ | ~~Medium~~ | ~~RESOLVED (Subsystem 6): `set_jet_job_mode()` on Phase wrapper + `Jet::map()`. 100/100 configurations converge.~~ | ~~Done~~ |
| Reentry | `refine_traj_manual` | Medium | Same gap as ParallelParking — not on Phase wrapper | (same fix) |
| ~~MinTimeToClimb~~ | ~~`InterpTable` in ODE~~ | ~~High~~ | ~~RESOLVED (Subsystem 4): `interp()` / `interp_scalar()` free functions + `InterpType` enum. Climb time 319.47 s matches Python ~321 s.~~ | ~~Done~~ |
| ~~MultiPhaseCannon~~ | ~~`add_direct_link_equal_con` on OCP wrapper~~ | ~~Medium~~ | ~~RESOLVED: 4 overloads on OCP wrapper~~ | ~~Done~~ |
| ~~MultiPhaseCannon~~ | ~~ODEParams region + named vars~~ | ~~Medium~~ | ~~RESOLVED: Phase wrapper region-aware resolution~~ | ~~Done~~ |
| ~~MultiPhaseCannon~~ | ~~Mixed state+ODE param inequality~~ | ~~Medium~~ | ~~RESOLVED: Phase wrapper mixed-variable overloads~~ | ~~Done~~ |
| DionysusLowThrust | MEE astro models | Medium | Python `MEETwoBody_CSI` and `CSIThruster` not available in C++. MEE dynamics must be implemented inline. | Expose MEE dynamics as C++ builder-compatible ODE factory |
| ~~DionysusLowThrust~~ | ~~`ODE::integrator()`~~ | ~~**High**~~ | ~~RESOLVED: 12 integrator overloads on ODE~~ | ~~Done~~ |
| ~~BettsLowThrust~~ | ~~Zonal gravity via `RowMatrix`/`inverse`~~ | ~~High~~ | ~~RESOLVED (Subsystem 6): Full J2/J3/J4 via MEEToCartesian (generated VF) + RowMatrix RTN rotation. Adaptive mesh converges.~~ | ~~Done~~ |
| ~~BettsLowThrust~~ | ~~MEE-to-Cartesian composition~~ | ~~Medium~~ | ~~RESOLVED (Subsystem 6): Uses generated `astro::MEEToCartesian` VF with analytic derivatives + `astro::MEEDynamics` for rate equations.~~ | ~~Done~~ |
| ~~OrbitContinuation~~ | ~~`ODE::integrator()`~~ | ~~**High**~~ | ~~RESOLVED: Same fix~~ | ~~Done~~ |
| ~~Heteroclinic~~ | ~~`LGLInterpTable` + `make_periodic()`~~ | ~~High~~ | ~~RESOLVED (Subsystem 5): LGLInterpTable already compatible with builder types. Full 4-stage port with STM manifolds + orbit-matching.~~ | ~~Done~~ |
| ~~Heteroclinic~~ | ~~`InterpFunction`~~ | ~~High~~ | ~~RESOLVED (Subsystem 5): `lgl_interp()` convenience function added. Used in orbit-matching constraints.~~ | ~~Done~~ |
| ~~Heteroclinic~~ | ~~STM integration (`integrate_stm_parallel`)~~ | ~~High~~ | ~~RESOLVED (Subsystem 5): DynIntegrator inherits all STM methods. Used for monodromy matrix eigendecomposition.~~ | ~~Done~~ |
| ~~OptimalDocking~~ | ~~`quat_rotate`~~ | ~~High~~ | ~~RESOLVED: `quat_rotate()` free function in VF DSL~~ | ~~Done~~ |
| ~~OptimalDocking~~ | ~~`cwise_product(Eigen::Vector)`~~ | ~~Medium~~ | ~~RESOLVED: Overload accepting Eigen vectors~~ | ~~Done~~ |
| ~~OptimalDocking~~ | ~~`InterpTable` for target trajectory~~ | ~~High~~ | ~~RESOLVED (Subsystem 5): Form2 implemented with LGLInterpTable + lgl_interp() for target trajectory. 6x faster than Form1.~~ | ~~Done~~ |
| MultiSpacecraftOpt | `PhasePack` link constraints | Low | Works but verbose. Must construct `PhasePack` tuples manually and use `ocp.base()`. | Add named-variable link constraint helpers to OCP wrapper |
| ~~MultiSpacecraftOpt~~ | ~~`sub_variables` on Phase wrapper~~ | ~~Medium~~ | ~~RESOLVED: Forwarded to Phase wrapper~~ | ~~Done~~ |
| ~~MinTimeClimbTables~~ | ~~`InterpTable1D`/`InterpTable2D` in ODE~~ | ~~High~~ | ~~RESOLVED (Subsystem 4): Same fix as MinTimeToClimb. Full table-based aero with `interp_scalar()` and `interp()`.~~ | ~~Done~~ |
| ~~GoddardRocket~~ | ~~Control-law integration for IG~~ | ~~**High**~~ | ~~RESOLVED: Control-law overloads on `ODE::integrator()`~~ | ~~Done~~ |
| ~~MultiPhaseCannon~~ | ~~Event detection in integrator~~ | ~~**High**~~ | ~~RESOLVED: `integrate_dense` with stop function~~ | ~~Done~~ |
