# Tycho Benchmark Comparison: pre-PR6 (rubber_types/C++17) vs PR6 (GFTypeErasure/C++20)

- **pre-PR6 (rubber_types/C++17)**: 2026-03-08 10:05:13
- **PR6 (GFTypeErasure/C++20)**: 2026-03-07 23:44:11

## Python Runtime

| Benchmark | pre-PR6 (rubber_types/C++17) (s) | PR6 (GFTypeErasure/C++20) (s) | Speedup (PR6 (GFTypeErasure/C++20)/pre-PR6 (rubber_types/C++17)) |
|-----------|---------------------------------|------------------------------|------------------------------------------------------------------|
| basic/bench_brachistochrone | 0.0048 ± 0.0010 | 0.0052 ± 0.0018 | 0.920× |
| basic/bench_bryson_denham | 0.0140 ± 0.0006 | 0.0141 ± 0.0007 | 0.996× |
| basic/bench_cartpole | 0.0303 ± 0.0005 | 0.0305 ± 0.0007 | 0.994× |
| basic/bench_dionysus | 0.2248 ± 0.0059 | 0.2192 ± 0.0026 | 1.026× |
| basic/bench_free_flying_robot | 0.2634 ± 0.0031 | 0.2599 ± 0.0037 | 1.014× |
| basic/bench_hanging_chain | 0.1961 ± 0.0064 | 0.1892 ± 0.0020 | 1.036× |
| basic/bench_mountain_car | 0.1120 ± 0.0116 | 0.1070 ± 0.0038 | 1.047× |
| basic/bench_optimal_docking | 1.1846 ± 0.0252 | 1.1388 ± 0.0050 | 1.040× |
| basic/bench_orbit_continuation | 0.2391 ± 0.0076 | 0.2307 ± 0.0067 | 1.037× |
| basic/bench_simple_lowthrust | 0.0665 ± 0.0020 | 0.0641 ± 0.0008 | 1.037× |
| basic/bench_topputto_lowthrust | 0.0925 ± 0.0018 | 0.0900 ± 0.0007 | 1.028× |
| basic/bench_vanderpol | 0.0141 ± 0.0002 | 0.0139 ± 0.0003 | 1.014× |
| basic/bench_zermelo | 0.0348 ± 0.0017 | 0.0336 ± 0.0011 | 1.036× |
| mesh_refinement/bench_hypersens | 0.0311 ± 0.0006 | 0.0308 ± 0.0013 | 1.007× |
| mesh_refinement/bench_minimum_time_climb | 6.6168 ± 0.0626 | 6.7546 ± 0.4831 | 0.980× |
| mesh_refinement/bench_reentry | 1.8453 ± 0.1574 | 1.8127 ± 0.1080 | 1.018× |
| multi_phase/bench_cannon | 0.2216 ± 0.0115 | 0.2255 ± 0.0029 | 0.983× |
| multi_phase/bench_delta3 | 1.2564 ± 0.0977 | 1.2863 ± 0.1079 | 0.977× |
| multi_phase/bench_goddard_rocket | 0.0468 ± 0.0044 | 0.0472 ± 0.0034 | 0.992× |
| multi_phase/bench_multi_phase_zermelo | 0.2157 ± 0.0041 | 0.2209 ± 0.0048 | 0.976× |

## C++ Runtime

| Benchmark | pre-PR6 (rubber_types/C++17) (s) | PR6 (GFTypeErasure/C++20) (s) | Speedup (PR6 (GFTypeErasure/C++20)/pre-PR6 (rubber_types/C++17)) |
|-----------|---------------------------------|------------------------------|------------------------------------------------------------------|
| brachistochrone_bench | 0.0075 ± 0.0005 | 0.0079 ± 0.0007 | 0.949× |
| gf_eval_bench | 0.1494 ± 0.0011 | 0.1422 ± 0.0022 | 1.051× |

## Notes

- **Speedup** > 1× means the second column is faster.
- **Reduction** positive % means the second column is smaller.
