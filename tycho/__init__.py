import inspect

import _tycho as _tycho

import tycho.Astro
import tycho.OptimalControl
import tycho.Solvers
import tycho.Utils
import tycho.VectorFunctions

SoftwareInfo = _tycho.SoftwareInfo

if __name__ == "__main__":
    _tycho.SoftwareInfo()
    mlist = inspect.getmembers(_asset)
    for m in mlist:
        print(m[0], "= _tycho." + str(m[0]))
