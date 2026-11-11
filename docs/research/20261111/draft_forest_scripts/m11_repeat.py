#!/usr/bin/env python3
"""Run the M10 reasoning repeatability subset while the main job is queued."""
from m11_measure import locked_group
for rep in range(10):
    locked_group('--repeat',rep)
