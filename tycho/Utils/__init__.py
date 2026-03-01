import inspect

import _tycho as _tycho
from _tycho.Utils import *

get_core_count = _tycho.Utils.get_core_count


if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.Utils)
    for m in mlist:
        print(m[0], "= _tycho.Utils." + str(m[0]))
