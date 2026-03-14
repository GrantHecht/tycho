# Tycho Benchmark Comparison: baseline vs main

- **baseline**: 2026-03-02 23:44:51
- **main**: 2026-03-07 21:34:18

## Compilation Time

| Metric | baseline | main | Speedup (main/baseline) |
|--------|----------|----------|---------|
| Full rebuild | 461.9 ± 0.0s | — | — |

## Python Runtime

| Benchmark | baseline (s) | main (s) | Speedup (main/baseline) |
|-----------|-------------|---------|-------------------------|
| benchmark_brachistochrone.py | 0.0053 ± 0.0034 | 0.0045 ± 0.0010 | 1.170× |
| benchmark_bryson_denham.py | 0.0124 ± 0.0006 | 0.0140 ± 0.0003 | 0.887× |
| benchmark_dionysus.py | 0.2111 ± 0.0008 | 0.2144 ± 0.0014 | 0.984× |
| benchmark_vanderpol.py | 0.0166 ± 0.0005 | 0.0127 ± 0.0002 | 1.312× |
| benchmark_zermelo.py | 0.0214 ± 0.0003 | 0.0215 ± 0.0001 | 0.999× |

## C++ Runtime

| Benchmark | baseline (s) | main (s) | Speedup (main/baseline) |
|-----------|-------------|---------|-------------------------|
| brachistochrone_bench | 0.0073 ± 0.0002 | 0.0076 ± 0.0005 | 0.963× |

## Binary Sizes

| Binary | baseline | main | Reduction |
|--------|----------|----------|-----------|
| `_tycho.so` | 60.45 MB | 38.01 MB | +37.1% |
| `brachistochrone_bench` | 11.55 MB | 9.91 MB | +14.1% |

## Notes

- **Speedup** > 1× means the second column is faster.
- **Reduction** positive % means the second column is smaller.
