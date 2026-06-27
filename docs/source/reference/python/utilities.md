(utilities-python)=
# Utilities

The Python API for Tycho's utilities subsystem, exposed through the
``tychopy.utils`` module. This page documents the **user-facing** subset:
thread-count management functions and the ``BumpAllocator`` arena interface.
Internal metaprogramming helpers are intentionally omitted.

```{eval-rst}
.. currentmodule:: tychopy.utils
```

## Threading

Functions for controlling the global thread pool used by Tycho's parallel
integration and optimal-control routines. Call ``set_num_threads`` once at
startup before any parallel computation begins.

```{eval-rst}
.. autofunction:: set_num_threads

.. autofunction:: get_num_threads

.. autofunction:: get_core_count
```

## Memory management

### ``BumpAllocator``

Per-thread bump allocator that manages SIMD-aligned arena buffers for ODE
evaluation temporaries. The allocator self-tunes upward after overflow;
``resize`` lets you pre-size the arenas to avoid runtime reallocation in
performance-critical loops.

```{eval-rst}
.. autoclass:: BumpAllocator
   :members:
```
