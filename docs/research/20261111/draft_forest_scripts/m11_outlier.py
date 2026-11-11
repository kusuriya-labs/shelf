#!/usr/bin/env python3
"""Balanced repeats of the CSV prompt after a slow chain measurement."""
from m11_measure import locked_group
for rep in range(2):
    locked_group('--outlier',rep)
