#!/usr/bin/env python3
"""Disable transparent huge pages for a command and its descendants only."""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41
if len(sys.argv) < 2:
    raise SystemExit('usage: plamo_disable_thp.py command [args...]')
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    err = ctypes.get_errno()
    raise OSError(err, os.strerror(err))
os.execvp(sys.argv[1], sys.argv[1:])
