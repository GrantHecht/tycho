# Tycho Benchmark Comparison: pybind11 vs nanobind

- **pybind11**: 2026-03-02 23:44:51
- **nanobind**: 2026-03-02 23:23:06

## Compilation Time

| Metric | pybind11 | nanobind | Speedup (nanobind/pybind11) |
|--------|----------|----------|---------|
| Full rebuild | 461.9 ± 0.0s | 737.9 ± 0.0s | 0.626× |

## Python Runtime

| Benchmark | pybind11 (s) | nanobind (s) | Speedup (nanobind/pybind11) |
|-----------|-------------|-------------|-----------------------------|
| benchmark_brachistochrone.py | 0.0053 ± 0.0034 | 0.0053 ± 0.0029 | 1.002× |
| benchmark_bryson_denham.py | 0.0124 ± 0.0006 | 0.0138 ± 0.0004 | 0.902× |
| benchmark_dionysus.py | 0.2111 ± 0.0008 | 0.2131 ± 0.0011 | 0.991× |
| benchmark_vanderpol.py | 0.0166 ± 0.0005 | 0.0135 ± 0.0005 | 1.232× |
| benchmark_zermelo.py | 0.0214 ± 0.0003 | 0.0216 ± 0.0002 | 0.991× |

## C++ Runtime

| Benchmark | pybind11 (s) | nanobind (s) | Speedup (nanobind/pybind11) |
|-----------|-------------|-------------|-----------------------------|
| brachistochrone_bench | 0.0073 ± 0.0002 | 0.0075 ± 0.0003 | 0.968× |

## Binary Sizes

| Binary | pybind11 | nanobind | Reduction |
|--------|----------|----------|-----------|
| `_tycho.so` | 60.45 MB | 58.17 MB | +3.8% |
| `brachistochrone_bench` | 11.55 MB | 11.62 MB | -0.6% |

## Notes

- **Speedup** > 1× means the second column is faster.
- **Reduction** positive % means the second column is smaller.
