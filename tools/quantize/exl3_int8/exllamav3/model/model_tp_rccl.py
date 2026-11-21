import math
import os
import torch
import torch.distributed as dist
from datetime import timedelta
from ..util import log_tp


# Finite default timeout for the process group: rendezvous and every collective are
# bounded, so a failed peer surfaces as a timeout instead of a permanent hang.
# 180s leaves room for cold RCCL/JIT startup on a real model; EXL3_TP_TIMEOUT_S
# overrides this default for slower first-collective environments.
DEFAULT_TIMEOUT_S = 180.0
TIMEOUT_ENV = "EXL3_TP_TIMEOUT_S"


def _default_timeout_s() -> float:
    """
    Group timeout from the environment, falling back to DEFAULT_TIMEOUT_S. Invalid,
    non-finite or non-positive EXL3_TP_TIMEOUT_S values are rejected with a warning
    rather than silently installing an unbounded or zero timeout.
    """
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        print(f" !! RCCL backend: ignoring invalid {TIMEOUT_ENV}={raw!r}, using {DEFAULT_TIMEOUT_S}s")
        return DEFAULT_TIMEOUT_S
    return value


class TPBackendRCCL:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = 0,
        dist_backend: str | None = None,
        timeout_s: float | None = None,
        close_barrier_timeout_s: float | None = None,
    ):
        """
        Tensor-parallel communication backend implemented entirely on torch.distributed
        collectives. On HIP builds the backend name "nccl" is served by RCCL, so this is
        the ROCm TP transport.

        Unlike the CUDA TPBackendNCCL, nothing is delegated to the native pg_* shared-memory
        collectives: broadcast, all_reduce and both gather variants are complete torch.distributed
        operations here. master, uuid and shbuf_size are accepted for interface compatibility
        with the other backends (the model loader passes them to every backend type) but no
        shared-memory segments are created. The shared-arena host registration used to import
        module weights is HIP-translated independently of this backend and is unaffected.

        Lifecycle policy:
        - The process group timeout is finite: the EXL3_TP_TIMEOUT_S override when set to a
          positive finite number, else DEFAULT_TIMEOUT_S.
        - If construction fails after this backend created the default process group (e.g. an
          RCCL comm-init failure during warmup), the owned group is destroyed automatically
          before the exception propagates. An already-existing ("foreign") group is never
          destroyed, neither here nor in close().
        - An already-initialized default group is adopted only when it is exactly this group:
          same rank, same world size, and a backend serving the requested transport. Otherwise
          the constructor raises and the existing group is left intact.
        - close() destroys only the group created here, with no extra collective: the loader
          has already drained normal inference. Teardown errors propagate. Repeat close is a
          no-op. close_barrier_timeout_s is a deprecated, ignored compatibility argument.
        - The CPU helper process (device < 0) joins nothing: it initializes no process group,
          touches no accelerator and runs no collective; its only backend interactions are the
          run_cpu_reduce_jobs / end_cpu_reduce_jobs no-ops.

        Device IDs map to ranks by position in active_devices, which is ordered with the output
        device last (the in-process pseudo-worker) and may otherwise be arbitrary.

        Correctness notes:
        - all_reduce keeps the tensor's own dtype. Float32 reductions stay float32; the CUDA
          NCCL backend's bf16 round-trip is deliberately not reproduced.
        - contribution=False zeroes the contribution buffer before the reduction, so a rank
          that contributes nothing sends exact zeros regardless of whether its buffer held
          uninitialized or NaN data.
        - Noncontiguous tensors are reduced/broadcast through a contiguous staging buffer and
          copied back with copy_, which handles strided outputs.
        - gather/gather_small are matched point-to-point transfers between each contributor and
          the output rank (a local copy for the output rank's own slice), so ranks outside
          gather_devices never enter the operation. Each contributor sends its payload with one
          send, and the output rank receives in gather_devices order, so latency scales with the
          number of contributors, not the number of collectives: for TP2 this is a single
          send/recv pair per gather.
        """
        self.device = device
        self.output_device = output_device
        self.active_devices = []
        self.rank = -1
        self.world_size = 0
        self.timeout_s = float(timeout_s) if timeout_s is not None else _default_timeout_s()
        self.dist_backend = dist_backend or "nccl"
        # Ownership bookkeeping: close() and constructor-failure cleanup only ever tear down
        # a group that was created here. Both flags stay in their safe state if init below
        # raises, and a failure AFTER ownership is acquired destroys the owned group before
        # propagating, so no half-initialized global PG leaks into the process.
        self.pg_active = False
        self.closed = True

        if device < 0:
            log_tp(device, "RCCL init: skip CPU process")
            return

        self.active_devices = list(active_devices)
        self.world_size = len(self.active_devices)
        self.rank = self.active_devices.index(device)
        output_rank = self.active_devices.index(output_device)
        assert math.isfinite(self.timeout_s) and self.timeout_s > 0, \
            "RCCL backend timeout must be finite and positive"

        log_tp(device, f"RCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, "
                       f"output rank {output_rank}, backend {self.dist_backend}, init_method {init_method}")
        try:
            if not dist.is_initialized():
                print(f" -- RCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, "
                      f"init_method {init_method}, timeout {self.timeout_s}s")
                dist.init_process_group(
                    self.dist_backend,
                    rank = self.rank,
                    world_size = self.world_size,
                    init_method = init_method,
                    timeout = timedelta(seconds = self.timeout_s),
                )
                self.pg_active = True
            else:
                self._adopt_existing_group()
            self.closed = False
            self.mp_warmup_rccl()
        except BaseException:
            if self.pg_active:
                self.pg_active = False
                self.closed = True
                try:
                    dist.destroy_process_group()
                    log_tp(device, "RCCL init: destroyed the process group created here after failed init")
                except Exception as e:
                    # Keep the original failure as the raised exception, but never hide a
                    # failed teardown: the surviving group will surface in any later init.
                    log_tp(device, f"RCCL init: failed to destroy owned process group ({e})")
            raise


    def mp_warmup_rccl(self):
        """
        RCCL, like NCCL, initializes its internal state lazily on the first collective, which
        can take tens of seconds. Force that here, while every rank has just joined the group
        and no forward pass is in flight.
        """
        if self.world_size < 2:
            return
        print(f" -- RCCL warmup, device {self.device}, please wait...")
        if self.dist_backend == "nccl" and torch.cuda.is_available():
            x = torch.ones((6,), device = self.device)
        else:
            x = torch.ones((6,))
        dist.all_reduce(x)
        print(f" -- Finished RCCL warmup, device {self.device}")


    def close(self):
        if self.closed:
            return
        self.closed = True
        if not self.pg_active:
            # Never created a group (failed before ownership) or adopted an existing one:
            # a group not owned here must survive close intact.
            return
        self.pg_active = False
        # No drain barrier: the loader has already drained normal inference, and a
        # teardown barrier can hang unboundedly after a worker failure. Destroy errors
        # propagate so teardown failures stay visible.
        dist.destroy_process_group()


    def _adopt_existing_group(self):
        """
        A default process group that was already initialized when this backend is
        constructed belongs to someone else (another component, or a TP split loaded
        earlier in this process). Adopt it only if it is exactly this group for us:
        same rank, same world size, and a backend string that serves the requested
        transport (plain "nccl" or "gloo", or a compound "cpu:gloo,cuda:nccl").
        On any mismatch raise and leave the existing group untouched; on adoption,
        ownership stays with the initializer, so close() will not destroy it.
        """
        backend = str(dist.get_backend()).lower()
        wanted = self.dist_backend.lower()
        components = {p.rsplit(":", 1)[-1].strip() for p in backend.split(",") if p.strip()}
        compatible = backend == wanted or wanted in components
        if not (
            dist.get_world_size() == self.world_size
            and dist.get_rank() == self.rank
            and compatible
        ):
            raise RuntimeError(
                f"RCCL backend: cannot join the already-initialized default process group "
                f"(backend {backend!r}, rank {dist.get_rank()}, world {dist.get_world_size()}) "
                f"with this split (backend {self.dist_backend!r}, rank {self.rank}, "
                f"world {self.world_size}). The existing group is left untouched."
            )
        log_tp(self.device, "RCCL init: adopting existing process group (close() will not destroy it)")


    def fwd_barrier(self):
        self._check_live("fwd_barrier")
        if self.world_size > 1:
            dist.barrier()


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self._check_live("broadcast")
        src_rank = self._rank_of(src_device)
        buf = self._stage_in(tensor)
        self._wait(dist.broadcast(buf, src = src_rank))
        self._stage_out(tensor, buf)


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # Every rank of the TP split enters this collective every time, in the same sequence
        # (the loader's contract); contribution=False means "sum the others, not me". The
        # tensor's dtype is passed to the collective unchanged: float32 stays float32.
        self._check_live("all_reduce")
        if self.world_size <= 1:
            if not contribution:
                tensor.zero_()
            return
        buf = self._stage_in(tensor)
        if not contribution:
            buf.zero_()
        self._wait(dist.all_reduce(buf))
        self._stage_out(tensor, buf)


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self._gather(tensor, out_tensor, gather_devices, out_device, ldims, "gather")


    def gather_small(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self._gather(tensor, out_tensor, gather_devices, out_device, ldims, "gather_small")


    def run_cpu_reduce_jobs(self):
        # CPU-assisted reduction is a native-backend mechanism. RCCL all-reduce runs on the
        # GPU ranks and the CPU helper participates in nothing, so there is no work here.
        pass


    def end_cpu_reduce_jobs(self):
        pass


    def _check_live(self, op: str):
        if self.rank < 0:
            raise AssertionError(f"RCCL backend: {op} called from the CPU helper process")
        if self.closed:
            raise AssertionError(f"RCCL backend: {op} called after close()")


    def _rank_of(self, device: int) -> int:
        try:
            return self.active_devices.index(device)
        except ValueError:
            raise ValueError(
                f"RCCL backend: device {device} is not part of this TP split {self.active_devices}"
            ) from None


    def _stage_in(self, tensor: torch.Tensor) -> torch.Tensor:
        # torch.distributed collectives operate on contiguous storage. A noncontiguous
        # tensor is staged into a contiguous buffer; _stage_out copies the result back
        # with copy_, which is also what writes strided output slices correctly.
        if tensor.is_contiguous():
            return tensor
        buf = torch.empty(
            tensor.shape,
            dtype = tensor.dtype,
            device = tensor.device,
            memory_format = torch.contiguous_format,
        )
        buf.copy_(tensor)
        return buf


    def _stage_out(self, tensor: torch.Tensor, buf: torch.Tensor):
        if buf is not tensor:
            tensor.copy_(buf)


    def _wait(self, work):
        # Sync-mode returns are not always Work objects: dist.send yields None and
        # dist.recv yields the resolved source rank; both already blocked until
        # completion. Only genuine Work handles need an explicit wait.
        if work is not None and hasattr(work, "wait"):
            work.wait()


    def _gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices,
        out_device: int,
        ldims: list[int],
        op: str,
    ):
        self._check_live(op)
        if gather_devices is None:
            raise AssertionError(f"RCCL backend: {op} requires the list of participating device IDs")
        devices = [int(d) for d in gather_devices]
        widths = [int(m) for m in ldims]
        assert len(devices) == len(widths), \
            f"{op}: gather_devices and ldims must have the same length"
        assert all(w >= 0 for w in widths), \
            f"{op}: ldims must be nonnegative, got {ldims}"
        assert len(set(devices)) == len(devices), \
            f"{op}: duplicate gather_devices in {devices}"
        for dev in devices + [out_device]:
            self._rank_of(dev)  # validates membership with a clear error
        out_rank = self._rank_of(out_device)

        # Contributor slice geometry is validated before any transfer so a mismatched
        # rank fails locally instead of desynchronizing the point-to-point sequence.
        # Leading dims (everything but the concatenated width) and dtype must match
        # the wire layout; the width itself must equal the declared ldim exactly, so
        # the local copy_ can never implicitly broadcast.
        if self.rank == out_rank:
            assert out_tensor is not None, \
                f"{op}: Output device must supply output tensor"
            assert out_tensor.shape[-1] == sum(widths), \
                f"{op}: Output tensor must match size of concatenated slices: {sum(widths)}"
            own_width = widths[devices.index(self.device)] if self.device in devices else 0
            if own_width > 0:
                self._check_slice_shape(tensor, out_tensor, own_width, op)
            offset = 0
            for src_device, width in zip(devices, widths):
                if width > 0:
                    dst = out_tensor.narrow(-1, offset, width)
                    if src_device == self.device:
                        dst.copy_(tensor)
                    else:
                        buf = torch.empty(dst.shape, dtype = dst.dtype, device = dst.device)
                        self._wait(dist.recv(buf, src = self._rank_of(src_device)))
                        dst.copy_(buf)
                offset += width
        else:
            if self.device not in devices:
                raise AssertionError(
                    f"{op}: rank device {self.device} is neither the output device nor a contributor"
                )
            width = widths[devices.index(self.device)]
            assert tensor.shape[-1] == width, \
                f"{op}: contributor width {tensor.shape[-1]} != expected {width}"
            if width > 0:
                buf = self._stage_in(tensor)
                self._wait(dist.send(buf, dst = out_rank))


    def _check_slice_shape(self, tensor: torch.Tensor, out_tensor: torch.Tensor, width: int, op: str):
        assert tensor.shape[-1] == width, \
            f"{op}: output rank tensor width {tensor.shape[-1]} != expected {width}"
        assert tensor.shape[:-1] == out_tensor.shape[:-1], \
            f"{op}: output rank tensor leading shape {tuple(tensor.shape[:-1])} " \
            f"!= output leading shape {tuple(out_tensor.shape[:-1])}"
        assert tensor.dtype == out_tensor.dtype, \
            f"{op}: output rank tensor dtype {tensor.dtype} != output dtype {out_tensor.dtype}"
