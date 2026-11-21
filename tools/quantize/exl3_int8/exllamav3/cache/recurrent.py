from collections import OrderedDict
import os

from ..constants import PAGE_SIZE
from ..util.memory import malloc_trim

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0

# Opt-in batched prune (EXL3_BATCH_RECURRENT_PRUNE=1; default off keeps the legacy per-entry
# path for fair A/B). When on, prune_stranded() collects every stranded checkpoint first and
# sends one picklable bulk-delete per rank instead of one TP dispatch per entry, and memory
# release is aggregated so malloc_trim only ever fires after the strong references to the
# dropped states are gone. tp_run captures the environment, so runs stay traceable.
_BATCH_PRUNE_ENV = "EXL3_BATCH_RECURRENT_PRUNE"

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
        }


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


    def put(self, key, state):
        """
        Add state to cache
        """
        if key in self:
            self.move_to_end(key)
        else:
            stashed_state = state.stash()
            state_size = stashed_state["checkpoint_size"]
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                pt = self.pagetable

                # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
                # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
                # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
                # extra cost, since the missing pages force a replay past this position either way.
                popped_key = None
                if pt is not None:
                    for k in self:
                        if not pt.is_resumable(k):
                            popped_key = k
                            break
                if popped_key is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_stranded"] += 1
                else:
                    popped_key, popped = self.popitem(last = False)
                    if pt is not None:
                        page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                        if page is not None and page.kv_position == PAGE_SIZE:
                            self.metrics["stash_evictions_live_kv"] += 1

                self.metrics["stash_evictions"] += 1
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

            self[key] = stashed_state
            self.update_total_size()


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.

        With EXL3_BATCH_RECURRENT_PRUNE=1 the work is aggregated: one bulk delete dispatch per rank holding
        every deduplicated handle, and the byte accounting (parent-side for the local path, worker-side for
        TP) happens once the dropped states' strong references are released. Return count and metrics are
        unchanged from the legacy path; put/eviction keeps dispatching per entry.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k in self if not self.pagetable.is_resumable(k)]
        if not stranded:
            return 0

        if os.environ.get(_BATCH_PRUNE_ENV, "0") != "1":
            for k in stranded:
                popped = self.pop(k)
                self.metrics["stash_pruned"] += 1
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
            self.update_total_size()
            return len(stranded)

        # Batch path. Pop first so the resumable-anchored entries, and any handle that a survivor still
        # shares with a stranded alias, remain visible while deciding what workers may delete: two keys can
        # reference the same stashed dict/handle, and dropping the stranded alias must not pull the handle
        # out from under a live checkpoint.
        popped = [self.pop(k) for k in stranded]
        self.metrics["stash_pruned"] += len(stranded)
        if self.model.loaded_tp:
            # Parent entries are metadata only under TP; the worker ranks own the host tensors and do
            # their own note_freed accounting, so the parent must not add the reported bytes here.
            live = {v["tp_handle"] for v in self.values()}
            handles = []
            seen = set()
            for st in popped:
                h = st["tp_handle"]
                if h in live or h in seen:
                    continue
                seen.add(h)
                handles.append(h)
            del popped, live, seen, st
            if handles:
                self.model.tp_dispatch_all(mp_cache_recurrent_del_bulk, (id(self), handles))
        else:
            # Local path: the popped dicts hold the actual state until released, so aggregate the bytes
            # (deduplicated by object identity, like update_total_size, and excluding any dict a
            # surviving key still references), drop the references, and only then account the release
            # so any resulting malloc_trim sees the memory as freed.
            live = {id(v) for v in self.values()}
            seen = set()
            total = 0
            for st in popped:
                i = id(st)
                if i in live or i in seen:
                    continue
                seen.add(i)
                total += st["checkpoint_size"]
            del popped, live, seen, st
            note_freed(total)
        self.update_total_size()
        return len(stranded)


    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    note_freed(_stashed_bytes(stashed))


def mp_cache_recurrent_del_bulk(local_context: dict, cache_id: int, cp_handles: list):
    """
    Batched counterpart of mp_cache_recurrent_del for a whole prune_stranded(): pop every selected handle,
    sum the actual host bytes, release the local strong references, then account the release in a single
    note_freed call. The legacy per-entry shape keeps each popped stash bound while note_freed runs, so a
    threshold-triggered malloc_trim can fire while the worker still pins the very pages it asks the OS to
    return; here the trim only ever sees memory that is genuinely unreachable. Preconditions are checked
    before any pop so a bad handle list cannot leave the worker cache partially deleted.
    """
    recurrent_cache = local_context["recurrent_cache"]
    missing = [h for h in cp_handles if h not in recurrent_cache]
    assert not missing, f"bulk recurrent delete: handles {missing} not in worker cache of {cache_id}"
    assert len(set(cp_handles)) == len(cp_handles), \
        f"bulk recurrent delete: duplicate handles in {cp_handles}"
    stashed = [recurrent_cache.pop(h) for h in cp_handles]
    total = sum(_stashed_bytes(s) for s in stashed)
    del stashed
    note_freed(total)
