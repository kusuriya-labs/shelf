"""RAM-only compressed-tile orders; GPU consumers always receive canonical tiles."""


def select_tile_order(extension, requested = "auto", enabled = True):
    if requested not in ("auto", "0", "1", "2"):
        raise ValueError("EXL3_MOE_CPU_TILE_ORDER must be auto, 0, 1 or 2")
    if not enabled:
        return 0
    if requested == "auto":
        if extension.exl3_moe_cpu_has_avx512_bw():
            return 1
        return 2 if extension.exl3_moe_cpu_has_avx2() else 0
    order = int(requested)
    if order == 1 and not extension.exl3_moe_cpu_has_avx512_bw():
        raise ValueError("band tile order requires an AVX-512BW tier")
    return order


def projection_tile_order(order, bits):
    # Only the existing band format exempts K8. N-major includes every width.
    return 0 if order == 1 and bits == 8 else order


def copy_trellis(dst, src, order):
    """Copy canonical weights into an existing arena view, also used by expert swaps."""
    tk, tn, ps = src.shape
    order = projection_tile_order(order, ps // 16)
    if order == 2:
        dst.view(tn, tk, ps).copy_(src.permute(1, 0, 2))
    elif order == 1:
        dst.view(tn // 8, tk, 8, ps).copy_(src.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3))
    elif order == 0:
        dst.copy_(src)
    else:
        raise ValueError("CPU MoE tile order must be 0, 1 or 2")
