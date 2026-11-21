# SPDX-License-Identifier: MIT
"""Optional host-only bridge for an already validated EXL3 native installation."""


def ensure_group_backend(ext):
    if hasattr(ext, 'hgemm_exact_group'):
        return True
    try:
        import exl3_moe_group_ext as bridge
    except (ImportError, OSError):
        return False
    try:
        bridge.configure_engine(ext.__file__)
    except RuntimeError:
        # A different native ABI must retain the ordinary inference path.
        return False
    ext.hgemm_exact_group = bridge.hgemm_exact_group
    ext._moe_exact_group_backend = bridge.__file__
    return True
