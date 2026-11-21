"""
Opt-in host-RAM residency guarantee for the n-gram (Engram) tables: EXL3_NGRAM_MLOCK=1.

The V620 TP2 benchmark holds the whole Engram table (~30 GiB) in the TP owner rank's host
RAM, and rocm_tools/rdna2/tp_run.py fails its audit unless every table page is mincore
resident before AND after inference. Host memory pressure was observed paging table pages
out after the load (7907144/7968789 pages resident post-run) even though the table never
streams from disk. mlock(2) on the table's own pages is what stops the kernel from evicting
them.

What this deliberately is NOT:
  * not cudaHostRegister / pin_memory: there is no copy, no GPU mapping, nothing imported
    from torch - this pins existing host pages where they already are;
  * not mlockall(): exactly the page ranges of the tensors named by the caller are locked,
    and the returned MlockedRanges explicitly owns that lock (unlock is only possible
    through the object that locked it);
  * not a limits changer: RLIMIT_MEMLOCK is only READ, to make a failure legible. Root or
    the process launcher arranges the memlock allowance (e.g. 'ulimit -l unlimited' before
    spawn); this module never touches resource limits.

Semantics: default off (only EXL3_NGRAM_MLOCK exactly "1" opts in), Linux only. When the
process opted in, a failed or unsupported lock raises MlockError instead of silently
continuing - silently continuing is precisely the unproven-residency state the benchmark
audit exists to reject. Unset the variable to load with paging allowed, as before.
"""

from __future__ import annotations

import ctypes
import os
import sys

ENV_VAR = "EXL3_NGRAM_MLOCK"


class MlockError(RuntimeError):
    """EXL3_NGRAM_MLOCK was requested but the residency guarantee cannot be given."""


def mlock_enabled() -> bool:
    """True iff EXL3_NGRAM_MLOCK=1 (default off; only the exact string "1" opts in).
    Read per call, never cached at import: the switch is opt-in and test-reconfigurable."""
    return os.environ.get(ENV_VAR, "0") == "1"


def _format_bytes(v: int) -> str:
    return f"{v / 2**20:.1f} MiB" if v >= (1 << 20) else f"{v} B"


def _memlock_hint() -> str:
    """Suffix for an mlock failure message: the limits this process can see, and what to do
    about them. Read-only - exllamav3 will not raise its own RLIMIT_MEMLOCK."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except Exception:
        return (f" [RLIMIT_MEMLOCK unreadable on {sys.platform!r}; this guarantee is "
                f"Linux-only, unset {ENV_VAR} to load without it]")
    inf = getattr(resource, "RLIM_INFINITY", -1)
    fmt = lambda v: "unlimited" if v == inf else _format_bytes(v)
    return (f" [RLIMIT_MEMLOCK soft={fmt(soft)} hard={fmt(hard)}: the process needs its "
            f"memlock allowance raised at launch (root arranges it: e.g. 'ulimit -l "
            f"unlimited' before spawning, or systemd LimitMEMLOCK); exllamav3 never "
            f"changes resource limits itself. Unset {ENV_VAR} to load without the "
            f"residency guarantee]")


_libc = None


def _load_libc() -> ctypes.CDLL:
    """The process libc with typed mlock/munlock (use_errno), cached. Linux-only, and
    failure to obtain either is an explicit MlockError, never a silent skip."""
    global _libc
    if _libc is None:
        if sys.platform != "linux":
            raise MlockError(
                f"{ENV_VAR}=1: the mlock-based RAM residency guarantee is Linux-only, this "
                f"host reports {sys.platform!r}. Unset {ENV_VAR} to load without it.")
        try:
            lib = ctypes.CDLL("libc.so.6", use_errno = True)
        except OSError as e:
            raise MlockError(
                f"{ENV_VAR}=1: could not open the C library for mlock/munlock: {e}") from e
        # correct signatures matter: c_void_p from int pointers, size_t lengths, int
        # returns, and errno captured per-thread (use_errno=True above)
        lib.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.mlock.restype = ctypes.c_int
        lib.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.munlock.restype = ctypes.c_int
        _libc = lib
    return _libc


def merged_page_ranges(ranges, page_size: int) -> list[tuple[int, int]]:
    """
    (address, nbytes) spans -> the minimal sorted list of page-aligned (start, length)
    ranges covering them: round heads/tails to page bounds, merge overlapping or touching
    runs. Every page ends up in exactly one returned range, so a table held as several
    tensors (or views sharing pages) costs one mlock per disjoint run and, crucially, the
    same ranges back out of unlock - no page is ever locked twice or unlocked that was not
    locked.
    """
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page size {page_size} is not a positive power of two")
    runs = []
    for addr, nbytes in ranges:
        addr, nbytes = int(addr), int(nbytes)
        if nbytes == 0:
            continue        # an empty tensor occupies no pages
        if addr <= 0:
            raise ValueError(f"invalid range to lock: address 0x{addr:x}, {nbytes} bytes")
        start = addr & -page_size
        end = (addr + nbytes + page_size - 1) & -page_size
        runs.append([start, end])
    runs.sort()
    merged = []
    for start, end in runs:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end - start) for start, end in merged]


def _call_libc(fn, name: str, addr: int, nbytes: int, what: str):
    """One mlock/munlock syscall: pointers/sizes checked before the call, errno (not just
    rc) reported after it, and the memlock-limit hint on any lock failure."""
    if addr <= 0 or nbytes <= 0:
        raise MlockError(f"{name} refused for {what}: bad range addr=0x{addr:x}, {nbytes} B")
    ctypes.set_errno(0)
    rc = fn(addr, nbytes)
    if rc != 0:
        err = ctypes.get_errno()
        detail = f"errno {err} ({os.strerror(err)})" if err else f"rc={rc}"
        msg = f"{name}(0x{addr:x}, {nbytes} B) failed for {what}: {detail}"
        raise MlockError(msg + (_memlock_hint() if name == "mlock" else ""))


class MlockedRanges:
    """
    The lock on a set of mlock(2)-ed page ranges: the merged, page-aligned spans of one or
    more contiguous CPU tensors, created locked-or-not-at-all, released idempotently.

    Explicit ownership of the lock lifetime: the object keeps a reference to every tensor it
    locked for exactly as long as any range is locked, so the munlock always addresses the
    very pages it mlocked - never memory the allocator has since handed to somebody else.
    No mlockall(), no global state, no implicit release path other than this object.
    """

    def __init__(self, libc = None, page_size: int | None = None):
        self._libc = libc                # None -> process libc at first syscall
        self._page_size = page_size      # None -> resource.getpagesize()
        self._locked: list[tuple[int, int]] = []
        self._tensors: tuple = ()
        self._what = "host memory"

    def __del__(self):
        # Backstop for a lock that was dropped without unlock(): its own tensor
        # references keep the ranges valid, so the pages can be returned here instead of
        # staying pinned while the allocator recycles them. Errors cannot be delivered to
        # anyone during GC (the kernel drops the locks at munmap/exit anyway).
        try:
            self.unlock()
        except Exception:
            pass

    @property
    def locked_ranges(self) -> list[tuple[int, int]]:
        return list(self._locked)

    @property
    def is_locked(self) -> bool:
        return bool(self._locked)

    @property
    def locked_bytes(self) -> int:
        return sum(n for _, n in self._locked)

    @staticmethod
    def _tensor_span(t, i: int, what: str) -> tuple[int, int]:
        """(data_ptr, nbytes) for a lockable tensor; anything that a single byte range
        cannot describe (non-host device, strided view, missing interface) is rejected
        explicitly rather than locked "best effort"."""
        dev = getattr(t, "device", None)
        if str(dev) != "cpu":
            raise MlockError(
                f"{what}: table tensor {i} is not host RAM (device {dev}); "
                f"{ENV_VAR} locks CPU tensors only")
        try:
            contiguous = bool(t.is_contiguous())
            ptr = int(t.data_ptr())
            nbytes = int(t.numel()) * int(t.element_size())
        except MlockError:
            raise
        except Exception as e:
            raise MlockError(
                f"{what}: table tensor {i} does not expose the tensor interface needed to "
                f"lock it: {e!r}") from e
        if not contiguous:
            raise MlockError(
                f"{what}: table tensor {i} is not contiguous; a strided view has no single "
                f"byte range to lock")
        if nbytes < 0 or ptr < 0:
            raise MlockError(
                f"{what}: table tensor {i} reports implausible storage "
                f"(data_ptr()=0x{ptr:x}, nbytes={nbytes})")
        if nbytes > 0 and ptr == 0:
            raise MlockError(f"{what}: table tensor {i} has no host address (data_ptr() == 0)")
        return ptr, nbytes

    def lock_tensors(self, tensors, what: str) -> MlockedRanges:
        """
        mlock the merged page spans of every tensor in `tensors` (all-or-nothing: when a
        range fails, the ranges already locked are munlocked before the MlockError escapes,
        and the caller keeps no half-taken lock). Returns self on success.
        """
        if self._locked:
            raise MlockError(
                f"{what}: this MlockedRanges already holds {len(self._locked)} locked "
                f"range(s); unlock() first (one object, one lock lifetime)")
        self._what = what
        tensors = list(tensors)
        if not tensors:
            raise MlockError(f"{what}: no table tensors to lock")
        spans = [self._tensor_span(t, i, what) for i, t in enumerate(tensors)]
        if self._libc is None:
            self._libc = _load_libc()       # validates platform + signatures, raises if unsupported
        page_size = self._page_size
        if page_size is None:
            import resource
            page_size = resource.getpagesize()
        ranges = merged_page_ranges(spans, page_size)
        if not ranges:
            raise MlockError(f"{what}: every table tensor is empty; nothing to lock")
        locked = []
        for start, length in ranges:
            try:
                _call_libc(self._libc.mlock, "mlock", start, length, what)
            except MlockError:
                # partial-lock rollback: adopt what we did get so unlock() owns it, run
                # the rollback through the same code path, and let the original failure
                # escape with anything the rollback could not undo attached to it
                self._locked = locked
                self._tensors = tuple(tensors)
                try:
                    self.unlock()
                except MlockError as rollback_error:
                    raise MlockError(
                        f"{what}: mlock failed and the partial lock could not be rolled "
                        f"back ({rollback_error})")
                raise
            locked.append((start, length))
        self._locked = ranges
        self._tensors = tuple(tensors)
        return self

    def unlock(self):
        """
        munlock every range this object still owns, then drop the tensor references.
        Idempotent: once fully unlocked (or if nothing was ever locked) this is a no-op, so
        repeated unload() passes are safe. If a munlock fails, the rest are still attempted
        and the failure raises: the failed ranges stay owned by this object (references
        held) so another unlock() can retry them - locked pages are never orphaned and
        never unlocked outside the ranges we locked.
        """
        if not self._locked:
            self._tensors = ()
            return
        if self._libc is None:
            self._libc = _load_libc()
        tried = len(self._locked)
        errors, remaining = [], []
        for start, length in reversed(self._locked):
            try:
                _call_libc(self._libc.munlock, "munlock", start, length, self._what)
            except MlockError as e:
                errors.append(str(e))
                remaining.append((start, length))
        remaining.reverse()
        self._locked = remaining
        if not remaining:
            self._tensors = ()      # nothing locked -> the tensors may go when the caller drops them
        if errors:
            raise MlockError(
                f"{self._what}: munlock failed on {len(errors)} of {tried} locked range(s): "
                f"{errors[0]}; {len(remaining)} range(s) remain owned by this MlockedRanges "
                f"(tensor references held) - call unlock() again to retry them")


def lock_tables_if_requested(tensors, what: str) -> MlockedRanges | None:
    """
    The shared entry point the n-gram RAM loaders call once their tables are materialized:
    returns an owned, acquired MlockedRanges when EXL3_NGRAM_MLOCK=1, or None when the
    switch is off (the default: not one syscall made, no state created). A raise means the
    guarantee was requested and cannot be given - callers must not continue as if resident.
    """
    if not mlock_enabled():
        return None
    return MlockedRanges().lock_tensors(tensors, what)
