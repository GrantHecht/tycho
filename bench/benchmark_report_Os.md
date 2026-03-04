# Tycho Benchmark Comparison: pybind11 vs nanobind-Os

- **pybind11**: 2026-03-02 23:44:51
- **nanobind-Os**: 2026-03-03 21:22:27

## Compilation Time

| Metric | pybind11 | nanobind-Os | Speedup (nanobind-Os/pybind11) |
|--------|----------|----------|---------|
| Full rebuild | 461.9 ± 0.0s | — | — |

## Python Runtime

| Benchmark | pybind11 (s) | nanobind-Os (s) | Speedup (nanobind-Os/pybind11) |
|-----------|-------------|----------------|--------------------------------|
| benchmark_brachistochrone.py | 0.0053 ± 0.0034 | 0.0051 ± 0.0025 | 1.041× |
| benchmark_bryson_denham.py | 0.0124 ± 0.0006 | 0.0138 ± 0.0003 | 0.899× |
| benchmark_dionysus.py | 0.2111 ± 0.0008 | 0.2188 ± 0.0010 | 0.965× |
| benchmark_vanderpol.py | 0.0166 ± 0.0005 | 0.0134 ± 0.0003 | 1.240× |
| benchmark_zermelo.py | 0.0214 ± 0.0003 | 0.0216 ± 0.0002 | 0.993× |

## C++ Runtime

| Benchmark | pybind11 (s) | nanobind-Os (s) | Speedup (nanobind-Os/pybind11) |
|-----------|-------------|----------------|--------------------------------|
| brachistochrone_bench | 0.0073 ± 0.0002 | 0.0078 ± 0.0006 | 0.934× |

## Binary Sizes

| Binary | pybind11 | nanobind-Os | Reduction |
|--------|----------|----------|-----------|
| `_tycho.so` | 60.45 MB | 49.02 MB | +18.9% |
| `brachistochrone_bench` | 11.55 MB | 11.62 MB | -0.6% |

## Notes

- **Speedup** > 1× means the second column is faster.
- **Reduction** positive % means the second column is smaller.
