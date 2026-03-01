import inspect

import _tycho as _tycho
from _tycho.Astro import *

Kepler = _tycho.Astro.Kepler
ModifiedDynamics = _tycho.Astro.ModifiedDynamics
cartesian_to_classic = _tycho.Astro.cartesian_to_classic
cartesian_to_classic_true = _tycho.Astro.cartesian_to_classic_true
cartesian_to_modified = _tycho.Astro.cartesian_to_modified
classic_to_cartesian = _tycho.Astro.classic_to_cartesian
classic_to_modified = _tycho.Astro.classic_to_modified
lambert_izzo = _tycho.Astro.lambert_izzo
modified_to_cartesian = _tycho.Astro.modified_to_cartesian
modified_to_classic = _tycho.Astro.modified_to_classic
propagate_cartesian = _tycho.Astro.propagate_cartesian
propagate_classic = _tycho.Astro.propagate_classic
propagate_modified = _tycho.Astro.propagate_modified


from .AstroFrames import CR3BPFrame

if __name__ == "__main__":
    mlist = inspect.getmembers(_tycho.Astro)
    for m in mlist:
        print(m[0], "= _tycho.Astro." + str(m[0]))
