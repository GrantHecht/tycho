(utilities-cpp)=
# Utilities

The C++ API for Tycho's utilities subsystem, declared in the
``tycho::utils`` namespace and exposed through the ``include/tycho/utils.h``
umbrella header. Python users should consult the
{doc}`Python reference </reference/python/utilities>`.

This page documents the **user-facing** subset: threading functions, the bump
allocator, a wall-clock timer, a flat associative map, compile-time factorial
helpers, and the type-name introspection utility. Internal metaprogramming
helpers (``SZ_*`` sizing expressions, tuple iterators, ``return_type_t``,
etc.) are intentionally omitted from this reference.

## Threading

Functions that configure and query the global thread pool used by Tycho's
parallel integration and optimal-control routines.

```{eval-rst}
.. doxygenfunction:: tycho::utils::set_num_threads
   :project: tycho

.. doxygenfunction:: tycho::utils::get_num_threads
   :project: tycho

.. doxygenfunction:: tycho::utils::get_core_count
   :project: tycho

.. doxygenfunction:: tycho::utils::parallel_blocks
   :project: tycho

.. doxygenfunction:: tycho::utils::parallel_sequence
   :project: tycho

.. doxygenfunction:: tycho::utils::parallel_task
   :project: tycho
```

## Memory management

### ``BumpAllocator``

Per-thread SIMD-aligned arena allocator for ODE evaluation temporaries.
Avoids heap allocation in inner loops by recycling fixed-capacity arena
buffers that self-tune upward after overflow.

```{eval-rst}
.. doxygenstruct:: tycho::utils::BumpAllocator
   :project: tycho
   :members:
```

## Timing

### ``Timer``

Lightweight wall-clock timer backed by ``std::chrono::high_resolution_clock``.
Useful for instrumenting integration and solver calls.

```{eval-rst}
.. doxygenclass:: tycho::utils::Timer
   :project: tycho
   :members:
```

## Containers

### ``FlatMap``

Sorted flat associative container (``std::vector``-backed). Provides
``O(log n)`` lookup with cache-friendly iteration; useful for small key–value
maps where insert order matters less than traversal speed.

```{eval-rst}
.. doxygenclass:: tycho::utils::FlatMap
   :project: tycho
   :members:
```

## Math helpers

Compile-time-friendly factorial utilities for combinatorial coefficients used
inside RK tableau and scaling computations.

```{eval-rst}
.. doxygenfunction:: tycho::utils::factorial
   :project: tycho

.. doxygenfunction:: tycho::utils::factorial_div
   :project: tycho
```

## Type introspection

```{eval-rst}
.. doxygenfunction:: type_name
   :project: tycho
```
