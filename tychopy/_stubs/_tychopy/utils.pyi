"""Contains miscilanaeous utilities"""

from typing import overload


def get_core_count() -> int: ...

def set_num_threads(n: int) -> None:
    """
    Set the number of threads for parallel work. n=1 for single-threaded mode, n>1 to use n threads.
    """

def get_num_threads() -> int:
    """Get the current thread count setting. 1 = single-threaded mode."""

class BumpAllocator:
    @overload
    @staticmethod
    def resize(arg: int, /) -> None: ...

    @overload
    @staticmethod
    def resize(arg0: int, arg1: int, /) -> None: ...

    @staticmethod
    def size_scalar() -> int: ...

    @staticmethod
    def size_super_scalar() -> int: ...
