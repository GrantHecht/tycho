# First-class variable bounds in PSIOPT — deferred, evidence path defined

Date: 2026-07-24. Status: **not scheduled; awaiting measurement** (see "How this gets
decided" below).

## Current state

Tycho has no variable-bound concept at the NLP level. `add_lower_var_bound` /
`add_upper_var_bound` and friends are user-level sugar that lower to ordinary
inequality constraints (`g(x) <= 0`) through the same path as any other inequality.
Each bound therefore costs one slack variable plus one inequality multiplier — two
KKT dimensions and a unit-coefficient coupling row — and for a bound constraint the
slack is a shifted duplicate of the variable itself (`s = u - x_i`).

## What first-class bounds would change

A mature interior-point treatment (Ipopt-style) handles bounds with the log-barrier
on `x` directly: no slack, no constraint row; a diagonal `Sigma = X^{-1} Z`
contribution to the (1,1) KKT block, bound multipliers `z_L`/`z_U`, and the
fraction-to-boundary rule applied to `x` itself. On bound-heavy collocation problems
(control/state bounds at every node), this eliminates thousands of KKT dimensions.

## Why it is deferred

This is program-scale surgery, not a feature: the `[primals | slacks | eq | iq]`
KKT layout is baked into the sparsity assembly, the solver's compound-vector views,
fraction-to-boundary (the primal step limit is currently computed from the slacks),
the mu/complementarity accounting, restoration's elastic algebra, and every
globalization component — all of which carry bit-identity or corpus gates. Native
bounds also require push-to-interior initialization for user guesses at or outside
their bounds. Doing this without evidence of the payoff would invert the
evidence-first discipline the solver program runs on.

## How this gets decided

The Ipopt reference harness keeps the shared NLP *identical* by default (bounds stay
general inequalities on both sides, so neither solver is structurally advantaged).
A cheap follow-on arm — Ipopt-side only, a small addition to the adapter — detects
bound-shaped inequality rows and lifts them into Ipopt's native `x_L`/`x_U` instead.
The measured delta between the identical arm and the lifted arm, on the bound-heavy
corpus problems (the cartpole/tight-bounds class), is the direct measurement of what
native bound handling is worth on our workloads. If that delta is large, first-class
bounds gets scheduled as its own program with the evidence attached; if small, this
note records why it stays closed.
