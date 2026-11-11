#!/usr/bin/env python3
"""Repeat M10's 256-token reasoning N=10 condition ten times."""
from m11_measure import locked_group
for group in range(2):
    locked_group('--repeat256', group)
