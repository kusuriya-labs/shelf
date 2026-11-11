#!/usr/bin/env python3
"""Run the M10 reasoning repeatability subset while the main job is queued."""
from m11_measure import MODES, run
for rep in range(10):
    order=MODES[rep % 4:]+MODES[:rep % 4]
    for mode in order:
        run(2,mode,rep,'repeat')
