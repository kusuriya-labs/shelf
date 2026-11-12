import asyncio
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/apps/server'))
import shelf_server as server

class Compute:
    instance = None
    def __init__(self, *args, **kwargs):
        Compute.instance = self
        self.capabilities = SimpleNamespace(prefix_cache=False)
        self.closed = False
    async def start(self):
        pass
    async def generate(self, *args, **kwargs):
        raise server.ComputeError('capacity')
    def is_alive(self):
        return True
    async def shutdown(self, *args, **kwargs):
        self.closed = True

class Service:
    inference_budget_ceiling = 32768
    def __init__(self, **kwargs):
        self.compute = kwargs['compute']
    async def complete(self, chat):
        raise server.ChatServiceError('capacity', code='compute_error')

class Ready(Exception):
    pass

class Runner:
    def __init__(self, *args, **kwargs):
        pass
    async def setup(self):
        pass

class Site:
    def __init__(self, *args, **kwargs):
        pass
    async def start(self):
        raise Ready()

server.codec.load_processor = lambda path: object()
server.AsyncComputeClient = Compute
server.ChatService = Service
server.build_compute_argv = lambda opts: []
server.create_app = lambda *args: None
server.web.AppRunner = Runner
server.web.TCPSite = Site
with tempfile.TemporaryDirectory() as temp:
    config = {"model_type": "plamo3", "hidden_size": 2560, "num_hidden_layers": 24,
              "vocab_size": 107520, "quantization_config": {"quant_method": "exl3",
              "bits": 4.05, "head_bits": 8}}
    (Path(temp) / "config.json").write_text(json.dumps(config))
    opts = server.parse_args(['--model-dir', temp, '--max-concurrent-requests', '8'])
    opts.compute = 'unused'
    log = io.StringIO()
    try:
        with contextlib.redirect_stderr(log):
            asyncio.run(server._serve(opts))
    except Ready:
        pass
    else:
        raise AssertionError('server failed to reach listener setup')
    assert 'prefill warmup skipped: capacity' in log.getvalue(), log.getvalue()
    assert 'chat warmup skipped: capacity' in log.getvalue(), log.getvalue()
    assert not Compute.instance.closed

    class DeadCompute(Compute):
        def is_alive(self):
            return False

    server.AsyncComputeClient = DeadCompute
    try:
        asyncio.run(server._serve(opts))
    except server.ComputeError:
        assert Compute.instance.closed
    else:
        raise AssertionError('dead compute should abort startup')
print('warmup failures preserve a live compute process and reject a dead one')
