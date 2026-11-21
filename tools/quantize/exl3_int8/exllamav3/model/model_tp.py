from __future__ import annotations
import multiprocessing
from multiprocessing import Process, Pipe
from multiprocessing.reduction import ForkingPickler
from ..util import find_free_port
from .model_tp_alloc import TPAllocator
import os
from typing import Callable
from ..util.memory import touch_device_measure_vram
from ..util.progress import ProgressBar
from .config import Config
from ..util.misc import Cleanupper
from .model_tp_fn import *
import uuid
from ..util import log_tp, global_t0
from ..tokenizer.mm_embedding import send_embeddings

cleanupper = Cleanupper()
DISPATCH_TIMEOUT = 20

def _load_dispatch_timeout() -> float:
    """
    Per-worker-response timeout while loading the model, separately configurable from the
    inference DISPATCH_TIMEOUT: a single mp_model_append on a huge MoE block off slow storage
    can legitimately far exceed what a decode-step ack may take, so the load path gets its own
    (longer) bound. The inference default is untouched.
    """
    default = 180.0
    raw = os.environ.get("EXL3_TP_LOAD_TIMEOUT")
    if raw is None:
        return default
    try:
        value = float(raw)
        if value > 0 and value == value and value != float("inf"):
            return value
    except ValueError:
        pass
    print(f" !! Ignoring invalid EXL3_TP_LOAD_TIMEOUT={raw!r}, using {default}s")
    return default

LOAD_DISPATCH_TIMEOUT = _load_dispatch_timeout()

# Bounded worker reaping in destroy_tp_context: join, terminate, join again so terminated
# children are reaped instead of left as zombies. Both waits are finite.
CHILD_JOIN_TIMEOUT = 2
CHILD_TERMINATE_JOIN_TIMEOUT = 5

class Model_TPMixin:

    def __init__(self):
        self.mp_children = []
        self.mp_parent_conn = []
        self.mp_child_conn = []
        self.loaded_tp = False
        self.tp_output_device = None
        self.tp_producer = None
        self.tp_backend = None
        # Timeout in effect for worker result waits. Always DISPATCH_TIMEOUT except while
        # _load_tp runs, which raises it to LOAD_DISPATCH_TIMEOUT for its own dispatches
        # (cold JIT / huge module imports off slow storage need far more slack than a
        # decode-step ack).
        self.tp_dispatch_timeout = DISPATCH_TIMEOUT
        # Devices whose per-forward None acks are still in flight (see forward_tp), and a strong
        # reference to the dispatched args for that pass: pickling CPU tensors (e.g. exported
        # recurrent-state handles) moves their storages into torch shared-memory segments that
        # only live as long as the sender-side objects, and the children may not have read the
        # command yet when forward_tp returns
        self.tp_pending_acks = []
        self.tp_pending_refs = None

    def create_tp_context(self, tp_backend: str):
        """
        Create the tensor-parallel worker context.

        TP runs one Python process per participating CUDA device so each worker can own its CUDA context, loaded
        module shards and cache tensors independently. The selected output device is represented in the parent
        process by pseudo pipe/process objects, while the remaining devices are spawned with multiprocessing Pipes
        for command dispatch. A final CPU worker slot uses device -1 for backend helper work such as native
        CPU-based reductions. All workers receive the same backend description and shared-memory producer metadata,
        allowing large tensors and collectives to move through shared memory instead of being pickled over pipes.

        The output device is the TP "master" for this process: it runs synchronously in the main process through
        PseudoParentConn instead of in a spawned worker. _load_tp() keeps that device last in active_devices so
        fan-out dispatch reaches spawned workers before invoking the blocking pseudo-worker path.

        Lifecycle hardening: the atexit hook is installed before anything that can fail, and a partial spawn
        (SMProducer/Pipe/Process.start/PseudoParentConn init raising) tears down everything built so far through
        destroy_tp_context, which tolerates the half-constructed state and surfaces any teardown shortfall as a
        cleanup error attached to the original exception (the original error always propagates; nothing claims a
        clean teardown it did not finish). Bounded by design: the pseudo-worker's own process group is only
        reachable through the PseudoParentConn object AFTER its constructor returns - a constructor that fails
        before assignment is torn down by the backend's own constructor cleanup (model_tp_rccl /
        model_tp_backend), and collective/blocking teardown there is bounded by that backend's PG timeout
        (EXL3_TP_TIMEOUT_S), not by anything in this module.
        """
        log_tp(None, "Creating TP context")

        # Must use spawn method to avoid CUDA errors. Docs say this should always be set by __main__ but seems
        # to work okay here
        multiprocessing.set_start_method("spawn", force = True)
        torch.multiprocessing.set_sharing_strategy("file_system")

        # Backend args
        self.tp_backend = tp_backend
        master_addr = os.environ.get("EXLLAMA_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("EXLLAMA_MASTER_PORT", find_free_port())
        match tp_backend:
            case "nccl":
                # Master address and port for the process group
                backend_args = {
                    "type": tp_backend,
                    "init_method": f"tcp://{master_addr}:{master_port}",
                    "uuid": uuid.uuid4().hex,
                }
            case "native":
                backend_args = {
                    "type": tp_backend,
                    "init_method": f"tcp://{master_addr}:{master_port}",
                    "uuid": uuid.uuid4().hex,
                }
            case _:
                raise ValueError(f"Unkwown backend type: {tp_backend}")

        # Spawn child processes, each running the mp_model_worker function
        num_devices = max(self.active_devices) + 1
        assert not self.mp_children
        assert not self.mp_parent_conn
        assert not self.mp_child_conn
        self.mp_children: list = [None] * (num_devices + 1)
        self.mp_parent_conn: list = [None] * (num_devices + 1)
        self.mp_child_conn: list = [None] * (num_devices + 1)
        self.tp_pending_acks = []
        self.tp_pending_refs = None

        # Install exit hook to avoid child processes hanging if main process exits before unloading model.
        # Registered before anything that can fail: destroy_tp_context is idempotent and tolerates a
        # half-constructed context, so even a partial spawn below is reaped by this hook (directly, through
        # the except path, or at interpreter exit if the caller abandons the model).
        cleanupper.register_atexit(self.destroy_tp_context)

        try:
            self.tp_producer = SMProducer(buffer_size = 64 * 1024**2)

            for rank, device in enumerate(self.active_devices + [-1]):
                log_tp(None, f"Spawning child process: {device}")
                if self.tp_output_device == device:
                    self.mp_parent_conn[device] = PseudoParentConn(
                        device,
                        self.active_devices,
                        self.tp_output_device,
                        backend_args,
                        self.tp_producer,
                        global_t0
                    )
                    self.mp_child_conn[device] = PseudoChildConn()
                    self.mp_children[device] = PseudoChild()
                else:
                    self.mp_parent_conn[device], self.mp_child_conn[device] = Pipe()
                    child = Process(
                        target = mp_model_worker, args = (
                            self.mp_child_conn[device],
                            device,
                            self.active_devices,
                            self.tp_output_device,
                            backend_args,
                            self.tp_producer.export(),
                            global_t0
                        )
                    )
                    # Link before start(): if start() raises after the popen object exists, the
                    # teardown path must still see the process and be able to reap it
                    self.mp_children[device] = child
                    child.start()
        except BaseException as exc:
            # Partial-failure cleanup: close the pseudo-worker's process group, quit/terminate/join
            # the children that started, close every pipe, unlink the arena and drop the atexit hook.
            # destroy_tp_context resets all state, so the caller sees the original error and a later
            # create (or an abandoned model) starts from a clean slate. Teardown shortfalls are never
            # swallowed: they are attached to the original exception for context, which still wins.
            try:
                self.destroy_tp_context()
            except Exception as cleanup_err:
                errors = getattr(cleanup_err, "tp_teardown_errors", None)
                self._tp_attach_cleanup_errors(exc, list(errors) if errors else [repr(cleanup_err)])
            except BaseException as cleanup_err:
                self._tp_attach_cleanup_errors(exc, [repr(cleanup_err)])
            raise

        log_tp(None, "TP context created")


    def destroy_tp_context(self):
        """
        Destroy child processes (when unloading TP model or atexit)

        Idempotent and total-state-resetting: safe to call twice, safe to call on a
        half-constructed context (the create_tp_context partial-failure path relies on it), and
        safe to call with the pseudo conn or the shared-memory producer missing. Only real,
        owned, started children are joined/terminated, with bounded waits (join, terminate,
        join again so terminated children are reaped rather than left as zombies); the
        pseudo-worker (which is this process) is never terminated, only quit. Every connection
        that exists is closed, deferred forward acks and their pinned arg references are
        dropped, loaded_tp is reset so the object can be reloaded, and the atexit hook is
        unregistered.

        Teardown is best-effort and never lies: a failure on one step (worker join, pseudo
        backend close, producer unlink, connection close, a child that survived terminate) is
        recorded, ALL remaining resources are still released, and the collected errors are
        raised at the end as a single RuntimeError carrying them in .tp_teardown_errors. On any
        path the caller can take, the context is detached from self, so a retry (or a second
        destroy) sees a clean slate. tp_output_device is deliberately retained (unload_tp()
        clears it): _load_tp's failure path still consults it after teardown.
        """
        log_tp(None, "Destroying TP context")
        teardown_errors = []

        # Collect any deferred forward acks while the pipes are still installed, so quit
        # commands aren't interleaved with stale results. A failed drain means a worker died
        # or hung mid-pass - recorded, never swallowed, and not a leak: the reaping below
        # still runs.
        try:
            self.tp_drain_acks()
        except Exception as e:
            log_tp(None, "Exception draining deferred acks during destroy")
            teardown_errors.append(f"draining deferred acks: {e!r}")

        # Move the whole context out of self first: every step below operates on the
        # snapshot, and a second call (or the atexit hook racing an explicit unload) then
        # sees empty state and is a no-op. loaded_tp follows the state out, so a restart
        # is possible even when a later teardown step fails; tp_output_device is kept.
        parent_conns, self.mp_parent_conn = self.mp_parent_conn, []
        child_conns, self.mp_child_conn = self.mp_child_conn, []
        children, self.mp_children = self.mp_children, []
        producer, self.tp_producer = self.tp_producer, None
        output_device = self.tp_output_device
        self.loaded_tp = False
        self.tp_dispatch_timeout = DISPATCH_TIMEOUT

        # Destroy process group in child processes
        for device, (parent_conn, child) in enumerate(zip(parent_conns, children)):
            if parent_conn is None or child is None:
                continue
            if device == output_device or isinstance(child, PseudoChild):
                continue
            if getattr(child, "pid", None) is None or child.pid == os.getpid():
                continue  # never started, or not a process we own
            try:
                if child.is_alive():
                    log_tp(device, f"Closing backend, device {device}")
                    parent_conn.send("quit")
            except Exception as e:
                log_tp(device, f"Exception while closing backend, device {device}")
                teardown_errors.append(f"signalling quit to worker device {device}: {e!r}")

        # Destroy process group in main process. Called last since it blocks the main process
        pseudo_conn = None
        if output_device is not None and 0 <= output_device < len(parent_conns):
            pseudo_conn = parent_conns[output_device]
        if isinstance(pseudo_conn, PseudoParentConn):
            try:
                pseudo_conn.quit()
            except Exception as e:
                log_tp(output_device, f"Exception while closing pseudo-worker backend, device {output_device}")
                teardown_errors.append(f"pseudo-worker quit, device {output_device}: {e!r}")
                # quit() only reaches its close() after the backend teardown succeeded; make sure
                # the pinned consumer and local context still go away
                try:
                    pseudo_conn.close()
                except Exception as e:
                    log_tp(output_device, f"Exception while closing pseudo-worker, device {output_device}")
                    teardown_errors.append(f"pseudo-worker close, device {output_device}: {e!r}")

        # Join child processes (terminate if hung), close connections. Only started,
        # owned, real children are joined/terminated; join on a never-started Process
        # raises and terminate on anything that isn't ours would be worse than the leak.
        for device, (parent_conn, child_conn, child) in \
                enumerate(zip(parent_conns, child_conns, children)):
            is_pseudo = child is not None and isinstance(child, PseudoChild)
            if child is not None and not is_pseudo and device != output_device:
                pid = getattr(child, "pid", None)
                if pid is not None and pid != os.getpid():
                    log_tp(None, f"Attempting to destroy child, device {device}")
                    try:
                        child.join(timeout = CHILD_JOIN_TIMEOUT)
                        alive = child.is_alive()
                    except Exception as e:
                        log_tp(None, f"Exception joining child, device {device}")
                        teardown_errors.append(f"joining worker device {device}: {e!r}")
                        alive = True  # assume stuck: fall through to terminate
                    if alive:
                        log_tp(None, f"Terminating child, device {device}")
                        try:
                            child.terminate()
                        except Exception as e:
                            teardown_errors.append(
                                f"terminating worker device {device}: {e!r}, "
                                f"pid {pid} may still be running")
                        try:
                            child.join(timeout = CHILD_TERMINATE_JOIN_TIMEOUT)
                            if child.is_alive():
                                teardown_errors.append(
                                    f"worker device {device} (pid {pid}) survived terminate "
                                    f"and is still running")
                        except Exception as e:
                            teardown_errors.append(
                                f"reaping terminated worker device {device}: {e!r}")
            # Close every connection that exists, including the output-device slots. The
            # pseudo parent conn was closed by quit()/close() above; closing a
            # multiprocessing Connection twice is itself safe, each close guarded here so a
            # single failure can't leave later slots open.
            if parent_conn is not None and not isinstance(parent_conn, PseudoParentConn):
                try:
                    parent_conn.close()
                except Exception as e:
                    log_tp(None, f"Exception closing parent connection, device {device}")
                    teardown_errors.append(f"closing parent connection, device {device}: {e!r}")
            if child_conn is not None:
                try:
                    child_conn.close()
                except Exception as e:
                    log_tp(None, f"Exception closing child connection, device {device}")
                    teardown_errors.append(f"closing child connection, device {device}: {e!r}")
            log_tp(None, f"Closed connections, device {device}")

        self.tp_pending_acks = []
        self.tp_pending_refs = None

        if producer is not None:
            try:
                producer.close()
            except Exception as e:
                log_tp(None, "Exception closing TP shared-memory producer")
                teardown_errors.append(f"closing TP shared-memory producer: {e!r}")

        # Unregister exit hook
        cleanupper.unregister_atexit(self.destroy_tp_context)

        if teardown_errors:
            err = RuntimeError(
                f"TP context teardown incomplete, {len(teardown_errors)} problem(s): "
                + "; ".join(teardown_errors))
            err.tp_teardown_errors = teardown_errors
            log_tp(None, str(err))
            raise err
        log_tp(None, "Destroyed TP context")


    def _tp_attach_cleanup_errors(self, exc, errors):
        """
        Surface teardown shortfalls on an exception that is already propagating (create/load
        error paths) without replacing it: attached as a note (3.11+) and as a
        tp_cleanup_errors attribute for harnesses reading structured reports.
        """
        if not errors:
            return
        msg = "TP cleanup during error recovery reported: " + "; ".join(errors)
        try:
            exc.add_note(msg)
        except (AttributeError, TypeError):
            pass
        existing = getattr(exc, "tp_cleanup_errors", None)
        exc.tp_cleanup_errors = list(existing or []) + list(errors)


    def tp_drain_acks(self):
        """
        Collect deferred per-forward acks from child workers before touching the pipes or the shared
        input arena again. forward_tp/prefill_tp leave the child ranks' end-of-pass None results in
        flight so the main process can launch sampling work on the output device while the children
        finish their module walks; the acks must be in before the next dispatch reuses the pipes and
        before prepare_inputs_for_tp resets the arena the stragglers may still be reading from.
        Child exceptions from the deferred pass surface here.
        """
        pending, self.tp_pending_acks = self.tp_pending_acks, []
        try:
            for device in pending:
                r = self.tp_worker_result(device)
                assert r is None, "TP logic error"
        finally:
            # All children have consumed the deferred pass's command; shared storages may be
            # released. Cleared even when a drain errored (dead or crashed worker): holding the
            # refs afterwards leaks torch shared-memory segments with no reader left to blame.
            self.tp_pending_refs = None


    def _tp_dead_workers(self, devices):
        """
        Reasons why the spawned workers for the SELECTED `devices` cannot take part in a
        dispatch: dead, never started, or outright missing from a half-built context. A
        selected-but-absent worker is NOT healthy - the pseudo-worker would block inside a
        collective waiting on a rank that can never arrive - so it fails the preflight too.
        The in-process pseudo rank is not a spawned worker and is exempt.
        """
        children = self.mp_children
        if not children:
            # Selected devices with no context at all: every worker is missing
            return [(device, "missing (no TP context)") for device in devices
                    if device != self.tp_output_device]
        out = []
        for device in devices:
            if device == self.tp_output_device:
                continue
            child = children[device] if -len(children) <= device < len(children) else None
            if isinstance(child, PseudoChild):
                continue
            if child is None:
                out.append((device, "missing"))
                continue
            pid = getattr(child, "pid", None)
            if pid is None:
                out.append((device, "never started"))
                continue
            if pid == os.getpid():
                continue  # not a process we own; joining/killing it would hurt this one instead
            if not child.is_alive():
                out.append((device, "is no longer alive"))
        return out


    def _tp_dead_error(self, dead):
        detail = "; ".join(
            f"device {device} worker {reason}" for device, reason in dead)
        return RuntimeError(
            f"TP dispatch aborted: {detail}. The tensor-parallel model must be unloaded "
            f"and reloaded.")


    def _tp_require_workers_alive(self, devices):
        """
        Preflight that every spawned worker about to take part in the following dispatch is
        still alive, and raise an informative error if not.

        The output-device pseudo-worker executes its command synchronously in the main process
        and can block inside a backend collective or barrier until every peer rank arrives;
        a dead child would therefore hang the main process indefinitely (pipe polls bound the
        wait after the fact, but not a collective entered inline). Checking before fan-out
        turns that hang into an immediate, named failure. The check is a cheap is_alive()
        (a non-blocking waitpid) per spawned rank, so the normal all-alive path pays
        microseconds.
        """
        dead = self._tp_dead_workers(devices)
        if dead:
            raise self._tp_dead_error(dead)


    def tp_worker_dispatch_single(self, device, fn, args):
        """
        Dispatch single function call to child and get return value
        """
        self.tp_worker_dispatch(device, fn, args)
        return self.tp_worker_result(device)


    def tp_worker_dispatch(self, device, fn, args):
        """
        Dispatch function call to child
        """
        if device == self.tp_output_device:
            # A pseudo-worker send executes fn inline in the main process and may block inside a
            # collective with every peer rank, so this dispatch carries the full fan-out liveness
            # preflight; a spawned worker's dispatch only checks its own pipe's owner.
            self._tp_require_workers_alive((*self.active_devices, -1))
        else:
            self._tp_require_workers_alive([device])
        self.tp_drain_acks()
        conn = self.mp_parent_conn[device]
        try:
            conn.send((fn, args))
        except Exception as e:
            # A dead rank usually fails the write (broken pipe): name it instead of surfacing
            # a bare OSError; anything else propagates unchanged.
            dead = self._tp_dead_workers([device])
            if dead:
                raise self._tp_dead_error(dead) from e
            raise


    def tp_worker_result(self, device):
        """
        Await and return result from child function, and propagate any exceptions to main process
        """
        conn = self.mp_parent_conn[device]
        timeout = self.tp_dispatch_timeout
        try:
            if conn.poll(timeout):
                result = conn.recv()
            else:
                dead = self._tp_dead_workers([device])
                hint = f" ({dict(dead).get(device, 'unresponsive')})" if dead else ""
                raise TimeoutError(f"Timed out after {timeout}s waiting for worker on device {device}{hint}")
        except EOFError as e:
            raise RuntimeError(f"TP worker on device {device} died while a result was pending") from e
        if isinstance(result, Exception):
            raise result
        return result


    def tp_worker_dispatch_multi(self, active_devices: list[int], fn, args, dev_args: list | None = None):
        """
        Dispatch one function call to multiple workers without waiting for results.

        args are shared across devices; dev_args, when provided, supplies per-device argument suffixes matched by
        active_devices order. Callers normally pass self.active_devices, whose last entry is the in-process output
        device. That ordering matters because dispatching to the pseudo-worker executes the function immediately and
        can block on TP collectives; spawned workers must already have received the same command before that happens.

        All spawned workers in the fan-out set are prefighted for liveness before any command is sent, so a dead
        peer fails here instead of hanging the output-device pseudo-worker inside a collective.
        """
        self._tp_require_workers_alive(active_devices)
        self.tp_drain_acks()
        for idx, device in enumerate(active_devices):
            d_args = args
            if dev_args is not None:
                d_args = d_args + dev_args[idx]
            conn = self.mp_parent_conn[device]
            conn.send((fn, d_args))


    def tp_worker_wait_multi(self, active_devices: list[int]):
        """
        Wait for a previously dispatched multi-worker call and return results in device order.

        With the standard self.active_devices ordering this waits on child-process results before the in-process
        output device result. This mirrors dispatch order and avoids treating the synchronous pseudo-worker as an
        early rendezvous point while child workers are still undispatched or unread.
        """
        r = []
        for device in active_devices:
            r.append(self.tp_worker_result(device))
        return r


    def tp_worker_dispatch_wait_multi(self, active_devices: list[int], fn, args, dev_args: list | None = None):
        """
        Dispatch a function to multiple workers and wait for all corresponding results.

        For TP-wide calls, pass devices in self.active_devices order so the output-device pseudo-worker remains
        last for both dispatch and result collection.
        """
        self.tp_worker_dispatch_multi(active_devices, fn, args, dev_args)
        return self.tp_worker_wait_multi(active_devices)


    def tp_cache_page_copy(self, cache_id: int, from_page: int, to_page: int, num_tokens: int):
        # active_devices is ordered with the output-device pseudo-worker last, so all spawned workers receive the
        # copy command before the main process enters the synchronous pseudo-worker call.
        for device in self.active_devices:
            self.tp_worker_dispatch(device, mp_cache_page_copy, (
                cache_id,
                from_page,
                to_page,
                num_tokens
            ))
        for device in self.active_devices:
            self.tp_worker_result(device)


    def tp_cpu_cache_init(self, cache_ids: list[int], max_slots: int = 0):
        """
        Size of one whole cache page across every rank, in bytes, allocating each rank's slot pool if max_slots
        is given. Called once with max_slots = 0 to size a slot, then again once the budget is divided up.
        """
        sizes = self.tp_worker_dispatch_wait_multi(
            self.active_devices,
            mp_cpu_cache_init,
            (cache_ids, max_slots)
        )
        return sum(sizes)


    def tp_cpu_cache_store(self, cache_ids: list[int], slot: int, page_index: int):
        """
        Copy a cache page out to system RAM on every rank, stored under slot. Returns the number of stores that
        had to pin memory synchronously, summed over ranks, since that stall happens out of sight of the main
        process.
        """
        cold = self.tp_worker_dispatch_wait_multi(
            self.active_devices,
            mp_cpu_cache_store,
            (cache_ids, slot, page_index)
        )
        return sum(cold)


    def tp_cpu_cache_fetch(self, cache_ids: list[int], slot: int, page_index: int):
        """
        Copy a stored cache page back into a page slot on every rank
        """
        self.tp_dispatch_all(mp_cpu_cache_fetch, (cache_ids, slot, page_index))


    def tp_dispatch_all(self, func, args):
        """
        Run the same worker function on every active TP device and require all workers to complete.

        self.active_devices keeps the output device last. Since that device is executed synchronously in the main
        process, this order gives child workers a chance to enter any collective/barrier before the main process
        does.
        """
        for device in self.active_devices:
            self.tp_worker_dispatch(device, func, args)
        for device in self.active_devices:
            self.tp_worker_result(device)


    def tp_dispatch_master(self, func, args):
        """
        Run a worker function only on the TP output device and return its result.
        """
        self.tp_worker_dispatch(self.tp_output_device, func, args)
        r = self.tp_worker_result(self.tp_output_device)
        return r


    def tp_dispatch_lm_head_argmax(self, args, return_max: bool = False):
        """
        Compute argmax over a tensor-parallel sharded LM head.

        Each device that owns a non-empty LM-head slice computes local maximum values and vocabulary indices for
        its shard. The partial maxima are gathered to the output device, where the final winner is selected and the
        global token index is returned. With return_max, the winning logit value comes back alongside (the same
        value the layer-split path exports as draft confidence); it is already on the output device from the
        gather, so this adds no transfer or synchronization.

        Dispatch follows self.active_devices order so the output-device pseudo-worker, if participating, runs after
        the spawned workers have been sent their local argmax command.
        """
        ad = {}
        for device in self.active_devices:
            a, b, _ = self.plan[device]["lm_head"]
            if b > a:
                ad[device] = a

        if len(ad) == 1 and self.tp_output_device in ad:
            v, i = self.tp_worker_dispatch_single(
                self.tp_output_device,
                mp_model_forward_lm_head_argmax,
                args + (ad[self.tp_output_device], None, None, self.config.vocab_size)
            )
            return (i, v) if return_max else i

        gd = sorted(set(ad.keys()) | {self.tp_output_device})
        ldims = [1 if d in ad else 0 for d in gd]

        dispatched = []
        for device in self.active_devices:
            if device in gd:
                self.tp_worker_dispatch(
                    device,
                    mp_model_forward_lm_head_argmax,
                    args + (ad.get(device, -1), gd, ldims, self.config.vocab_size)
                )
                dispatched.append(device)

        results = []
        for device in dispatched:
            r = self.tp_worker_result(device)
            if r is not None:
                results.append((device, r))

        assert len(results) == 1 and results[0][0] == self.tp_output_device, \
            "TP logic error"

        device = self.tp_output_device
        vals, inds = [], []
        all_vals, all_inds = results[0][1]
        p = 0
        for d, ldim in zip(gd, ldims):
            if d in ad:
                vals.append(all_vals[..., p].to(device))
                inds.append(all_inds[..., p].to(device))
            p += ldim
        vals = torch.stack(vals, dim = -1)
        inds = torch.stack(inds, dim = -1)
        winner = vals.argmax(dim = -1, keepdim = True)
        argmax = inds.gather(-1, winner).squeeze(-1)
        if return_max:
            return argmax, vals.gather(-1, winner).squeeze(-1)
        return argmax


    # def tp_dispatch_lm_head_argmax_old(self, args):
    #     ad = []
    #     for device in self.active_devices:
    #         a, b, _ = self.plan[device]["lm_head"]
    #         if b > a:
    #             self.tp_worker_dispatch(device, mp_model_forward_lm_head_argmax_old, args + (a,))
    #             ad.append(device)
    #     results = []
    #     for device in ad:
    #         r = self.tp_worker_result(device)
    #         results.append((device, r))
    #
    #     device = self.tp_output_device
    #     vals, inds = [], []
    #     for (_, result) in sorted(results):
    #         v, i = result
    #         vals.append(v.to(device))
    #         inds.append(i.to(device))
    #     vals = torch.stack(vals, dim = -1)
    #     inds = torch.stack(inds, dim = -1)
    #     winner = vals.argmax(dim = -1)
    #     argmax = inds.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
    #     return argmax


    def _load_tp(
        self,
        progressbar: bool,
        reserve_per_device: list[int] | None,
        use_per_device: list[int] | None,
        active_devices: list[int],
        max_chunk_size: int,
        max_output_size: int,
        max_output_factor: int,
        callback_sync: Callable[[int, int], None],
        generator: bool,
        tp_output_device: torch.device | int | str,
        config: Config,
        modules: list,
        dev_limits: dict | None,
        tp_backend: str,
        verbose: bool,
        tp_options: dict,
    ):
        assert use_per_device is None or reserve_per_device is None
        if dev_limits is None: dev_limits = {}

        # Set output device
        if tp_output_device is None:
            tp_output_device = active_devices[0]
        self.tp_output_device = torch.device(tp_output_device).index

        # Move output device to end of active device list. This device is the TP "master" running synchronously in
        # the main process via PseudoParentConn, so keeping it last prevents fan-out helpers from blocking in the
        # main process before child workers have received their commands.
        active_devices.remove(self.tp_output_device)
        active_devices.append(self.tp_output_device)

        # Create TP context
        self.active_devices = active_devices
        self.create_tp_context(tp_backend)

        # From here on, everything this function builds (workers' module shards, the temporary
        # distribution pipeline, parent-side loaded modules and the config's deferred-load STC
        # bracket) must be torn down if the load raises OR the generator is abandoned mid-way
        # (GeneratorExit), otherwise a failed load leaks child processes, shared-memory segments
        # and a config stuck in deferred mode. create_tp_context already self-cleans on partial
        # failure, and destroy_tp_context is idempotent, so the cleanup below is total regardless
        # of how far we got.
        producer = None
        loaded_module = None
        defer_open = False
        try:
            # The load path gets its own (longer, separately configurable) worker-response
            # timeout: first-touch collectives and huge module imports can legitimately far
            # exceed what an inference ack may take. Inference keeps DISPATCH_TIMEOUT.
            self.tp_dispatch_timeout = LOAD_DISPATCH_TIMEOUT

            # Split model
            num_devices = max(self.active_devices) + 1
            max_mem = [0] * num_devices
            free_total = self.tp_worker_dispatch_wait_multi(self.active_devices, touch_device_measure_vram, ())
            for device, (free, total) in zip(self.active_devices, free_total):
                # print(free / 1024**3)  snip
                if reserve_per_device is not None:
                    free -= reserve_per_device[device]
                if use_per_device is not None:
                    free = use_per_device[device]
                max_mem[device] = free

            # Define TP split
            components = []
            for m in modules:
                components += m.make_tp_allocation(tp_options)
            order_keys = set((tp_options.get("expert_order") or {}).keys())
            unused = order_keys - {c.key for c in components if c.channel_unit == "experts"}
            if unused:
                raise ValueError(f"Expert placement contains unknown or unsupported MoE layers: {sorted(unused)}")
            allocator = TPAllocator(
                components,
                num_tokens = max_chunk_size,
                output_num_tokens = max_output_size,
                dev_limits = dev_limits,
            )
            allocator.initial_split(max_mem)
            if verbose:
                allocator.print_split()
            self.plan = allocator.compile_tp_plan()
            self.tp_worker_dispatch_wait_multi(self.active_devices, mp_set_plan, (self.plan, self.active_devices))

            # Distribution pipeline
            producer = SMProducer()
            self.tp_worker_dispatch_wait_multi(
                self.active_devices,
                mp_set_consumer,
                (),
                [(producer.export(),) if d != tp_output_device else (producer,) for d in active_devices]
            )

            # Begin loading modules
            with (ProgressBar(f"Loading (TP)" if progressbar else None, len(modules)) as progress):
                for idx, module in enumerate(modules):
                    last_module = module

                    if callback_sync: callback_sync(idx, len(modules))
                    if generator: yield idx, len(modules)

                    # Load module to CPU. tp_parent_defer lets payload-heavy modules that the plan
                    # places whole on a single TP rank (the n-gram Engram table under --ngram_ram)
                    # skip materializing in the parent: their tp_export ships the source locations
                    # and the owning rank loads the payload itself. Never a second copy, never
                    # through shared memory.
                    loaded_module = module
                    defer = module.can_defer_load()
                    if defer:
                        config.stc.begin_deferred_load()
                        defer_open = True
                    module.load(torch.device("cpu"), tp_parent_defer = True)
                    if defer:
                        config.stc.end_deferred_load()
                        defer_open = False

                    # Do module-specific device/process split
                    exported = module.tp_export(self.plan, producer)
                    self.tp_worker_dispatch_wait_multi(self.active_devices, mp_model_append, (exported,))
                    producer.clear()

                    # Release loaded module
                    module.unload()
                    loaded_module = None

                    # Progress and callbacks per fully loaded module
                    progress.update(idx + 1)

                # Append final gather layer
                if last_module.caps["logits_output"]:
                    self.tp_worker_dispatch_wait_multi(self.active_devices, mp_model_append_gather, ())

                # Final callback, 100% loaded
                if callback_sync: callback_sync(len(modules), len(modules))
                if generator: yield len(modules), len(modules)

            # Distribution pipeline
            self.tp_worker_dispatch_wait_multi(self.active_devices, mp_close_consumer, ())
            producer.close()
            producer = None

            config.stc.close()
            self.loaded_tp = True
        except BaseException as exc:
            # Best-effort teardown in dependency order, keeping the original error visible even
            # if a cleanup step also fails: every cleanup shortfall is attached to the original
            # exception (note + tp_cleanup_errors) instead of being swallowed.
            self.tp_dispatch_timeout = DISPATCH_TIMEOUT  # any destroy-side waits must be short
            cleanup_errors = []
            if defer_open:
                try:
                    config.stc.abort_deferred_load()
                except Exception as e:
                    cleanup_errors.append(f"aborting deferred load: {e!r}")
            if loaded_module is not None:
                try:
                    loaded_module.unload()
                except Exception as e:
                    cleanup_errors.append(f"unloading partially loaded module: {e!r}")
            if producer is not None:
                try:
                    producer.close()
                except Exception as e:
                    cleanup_errors.append(f"closing distribution producer: {e!r}")
            self.loaded_tp = False
            try:
                self.destroy_tp_context()
            except Exception as e:
                teardown = getattr(e, "tp_teardown_errors", None)
                cleanup_errors.extend(teardown if teardown else [f"destroying TP context: {e!r}"])
            except BaseException as e:
                cleanup_errors.append(f"destroying TP context: {e!r}")
            if cleanup_errors:
                log_tp(None, "Cleanup after failed TP load reported: " + "; ".join(cleanup_errors))
                self._tp_attach_cleanup_errors(exc, cleanup_errors)
            raise
        finally:
            self.tp_dispatch_timeout = DISPATCH_TIMEOUT

        if 'yield' in locals():
            yield


    def unload_tp(self):
        if not self.loaded_tp:
            return
        self.destroy_tp_context()
        self.loaded_tp = False
        self.tp_output_device = None
        cleanupper.unregister_atexit(self.destroy_tp_context)


    def prepare_inputs_for_tp(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        self.tp_producer.clear()
        # Use ID of Cache object as reference to avoid having to pickle it
        reserve = {}
        if "cache" in params:
            params["cache"] = id(params["cache"])
        # Share memory of any additional CPU tensors. Everything tensor-shaped must go through
        # the arena: CPU tensors pickled over the pipes become torch shared-memory segments with
        # fragile cross-process lifetimes (the recurrent_slots tensor used to crash GDN models
        # this way once forward acks became deferred)
        for tensor_param in [
            "block_table",
            "cache_seqlens",
            "positions",
            "position_ids",
            "recurrent_slots",
            "inv_freq",
            "input_ids",     # hash-MoE routing (DeepSeek-V4 bootstrap layers)
        ]:
            p = params.get(tensor_param)
            if p is not None:
                params[tensor_param] = self.tp_producer.send(p)

        p = params.get("indexed_embeddings")
        if p is not None:
            params["indexed_embeddings"] = send_embeddings(self.tp_producer, p)

        p = params.get("recurrent_states")
        if p is not None:
            reserve["recurrent_states"] = params["recurrent_states"]
            params["recurrent_states"] = [(rs.tp_export() if rs is not None else None) for rs in p]

        return self.tp_producer.send(x), reserve


    def restore_tp_params(self, params: dict, reserve: dict):
        # Read back state the forward pass wrote into the exported recurrent-state handles (the
        # output device's pseudo worker runs in this process, so its mutations are visible), then
        # restore the reserved originals
        exported_rs = params.get("recurrent_states")
        params.update(reserve)
        if "recurrent_states" in reserve and exported_rs is not None:
            for exp, orig in zip(exported_rs, reserve["recurrent_states"]):
                if exp is not None and orig is not None and hasattr(orig, "tp_readback"):
                    orig.tp_readback(exp)


    def prefill_tp(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        # Preflight every spawned rank (children and the CPU helper) up front: this method
        # feeds the child ranks through raw send_bytes and then enters the blocking, in-process
        # pseudo-worker forward, so a dead rank must fail here rather than hang the main process
        # inside a collective (or wedge on a pipe write to a corpse mid-fan-out).
        self._tp_require_workers_alive((*self.active_devices, -1))
        self.tp_worker_dispatch(-1, mp_cpu_reduce, ())

        x, reserve = self.prepare_inputs_for_tp(x, params)
        # active_devices order sends work to spawned CUDA workers first and the main-process output device last.
        # mp_model_forward enters backend barriers, so dispatching the synchronous pseudo-worker early would block
        # before the other ranks had even received this forward command. Child ranks all receive the same
        # command, pickled once and sent as raw bytes (Connection.send is dumps + send_bytes, so this is
        # wire-identical) — per-rank pickling of the params dict was part of the dispatch stagger.
        args = (x, params, last_kv_module_idx, True, None)
        msg = ForkingPickler.dumps((mp_model_forward, args))
        for device in self.active_devices:
            if device == self.tp_output_device:
                self.tp_worker_dispatch(device, mp_model_forward, args)
            else:
                self.mp_parent_conn[device].send_bytes(msg)
        # Same deferred-ack scheme as forward_tp: only the inline pseudo-worker's result is
        # consumed here, child acks drain at the next dispatch
        r = self.tp_worker_result(self.tp_output_device)
        assert r is None, "TP logic error"
        self.tp_pending_acks = [d for d in self.active_devices if d != self.tp_output_device]
        self.tp_pending_acks.append(-1)
        # See forward_tp: the exported recurrent-state handles must outlive the deferred acks
        self.tp_pending_refs = (args, params.get("recurrent_states"))
        self.restore_tp_params(params, reserve)
        return None


    def forward_tp(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        # Preflight every spawned rank (children and the CPU helper) up front: this method
        # feeds the child ranks through raw send_bytes and then enters the blocking, in-process
        # pseudo-worker forward, so a dead rank must fail here rather than hang the main process
        # inside a collective (or wedge on a pipe write to a corpse mid-fan-out). The pseudo
        # dispatch at the end of the loop re-checks, closing the window between fan-out and the
        # blocking call; each pass is just a waitpid per rank.
        self._tp_require_workers_alive((*self.active_devices, -1))
        self.tp_worker_dispatch(-1, mp_cpu_reduce, ())

        x, reserve = self.prepare_inputs_for_tp(x, params)
        # Keep the output-device pseudo-worker last for the same reason as prefill_tp(): its send() path executes
        # immediately in the main process and may block inside TP collectives until child workers arrive.
        # Shared pre-pickled command bytes for child ranks, as in prefill_tp
        args = (x, params, last_kv_module_idx, False, None)
        msg = ForkingPickler.dumps((mp_model_forward, args))
        for device in self.active_devices:
            if device == self.tp_output_device:
                self.tp_worker_dispatch(device, mp_model_forward, args)
            else:
                self.mp_parent_conn[device].send_bytes(msg)
        # The output-device pseudo-worker ran inline during the dispatch loop above, so its result
        # is already buffered. The child ranks' None acks (and the CPU helper's) are left in flight
        # and drained at the next dispatch, letting the caller queue sampling work on the output
        # device while the child processes finish their module walks
        out = self.tp_worker_result(self.tp_output_device)
        assert out is not None, "TP logic error"
        self.tp_pending_acks = [d for d in self.active_devices if d != self.tp_output_device]
        self.tp_pending_acks.append(-1)
        # Pin the exported recurrent-state handles too: restore_tp_params swaps them out of the
        # params dict in place, so holding args alone would not keep their shared storages alive
        self.tp_pending_refs = (args, params.get("recurrent_states"))
        self.restore_tp_params(params, reserve)
        return out


    def tp_rotate_cache_pages(self, cache_id: int, all_rotations: torch.Tensor):
        all_rotations = self.tp_producer.send(all_rotations)
        self.tp_worker_dispatch_wait_multi(self.active_devices, mp_rotate_cache_pages, (
            cache_id,
            all_rotations
        ))
