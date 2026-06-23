"""Contains miscilanaeous utilities"""

from typing import overload


def get_core_count() -> int:
    """
    Return the number of physical CPU cores on the current machine.

    Uses platform-specific APIs (e.g. ``sysconf`` on Linux/macOS,
    ``GetSystemInfo`` on Windows) to query the physical core count.  Falls
    back to ``std::thread::hardware_concurrency()`` when the platform query
    is unavailable or fails.

    Returns
    -------
    int
        Physical core count, or ``1`` if detection fails.
    """

def set_num_threads(n: int) -> None:
    """
    Set the number of threads used for parallel work.

    This is the process-global execution budget and is intended to be called
    once at startup, before any parallel computation begins.  Do not call it
    from inside a parallel region or from a worker thread.

    Parameters
    ----------
    n : int
        Thread count.  ``n <= 1`` selects single-threaded mode (all work
        runs inline on the calling thread; the thread pool is not used).
        ``n > 1`` sizes the pool to ``n`` worker threads.
    """

def get_num_threads() -> int:
    """
    Return the current process-global thread count.

    Returns
    -------
    int
        Number of threads.  ``1`` means single-threaded mode.
    """

class BumpAllocator:
    """
    Per-thread bump allocator for ODE evaluation temporaries.

    Tycho's ODE and VectorFunction evaluation stack uses thread-local
    SIMD-aligned arena buffers to avoid heap allocation in inner loops.
    This class exposes the management interface for those arenas.

    There are two per-thread arenas:

    * **Scalar arena** — holds ``double`` temporaries.
    * **SuperScalar arena** — holds ``DefaultSuperScalar`` (SIMD-width
      ``double`` array) temporaries.

    The default arena sizes are chosen automatically; use :meth:`resize` only
    when profiling reveals arena overflow (fallback heap allocation) in tight
    loops.
    """

    @overload
    @staticmethod
    def resize(arg: int, /) -> None:
        """
        Resize both per-thread arenas to the same element count.

        Must be called with no active allocations (i.e., outside any ODE or
        VectorFunction evaluation call).

        Parameters
        ----------
        size : int
            New capacity in elements for both the scalar and super-scalar arenas.
        """

    @overload
    @staticmethod
    def resize(arg0: int, arg1: int, /) -> None:
        """
        Resize the scalar and super-scalar arenas independently.

        Must be called with no active allocations (i.e., outside any ODE or
        VectorFunction evaluation call).

        Parameters
        ----------
        size_scalar : int
            New capacity in ``double`` elements for the scalar arena.
        size_super_scalar : int
            New capacity in ``DefaultSuperScalar`` elements for the
            super-scalar arena.
        """

    @staticmethod
    def size_scalar() -> int:
        """
        Return the current capacity of the per-thread scalar arena.

        Returns
        -------
        int
            Capacity in ``double`` elements.
        """

    @staticmethod
    def size_super_scalar() -> int:
        """
        Return the current capacity of the per-thread super-scalar arena.

        Returns
        -------
        int
            Capacity in ``DefaultSuperScalar`` elements.
        """
