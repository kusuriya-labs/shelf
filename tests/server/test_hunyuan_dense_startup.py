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
        return {'generated_ids': [1], 'finish_reason': 'stop'}

    def is_alive(self):
        return True

    async def shutdown(self, *args, **kwargs):
        self.closed = True


class Service:
    inference_budget_ceiling = 32768

    def __init__(self, **kwargs):
        self.compute = kwargs['compute']


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


def main():
    orig_load = server.codec.load_processor
    orig_probe = server.codec.reasoning_capability
    orig_client = server.AsyncComputeClient
    orig_service = server.ChatService
    orig_argv = server.build_compute_argv
    orig_app = server.create_app
    orig_runner = server.web.AppRunner
    orig_site = server.web.TCPSite

    def fail_probe(*args, **kwargs):
        raise AssertionError('hunyuan_v1_dense must not probe reasoning capability')

    server.codec.load_processor = lambda path, **kwargs: object()
    server.codec.reasoning_capability = fail_probe
    server.AsyncComputeClient = Compute
    server.ChatService = Service
    server.build_compute_argv = lambda opts: []
    server.create_app = lambda *args: None
    server.web.AppRunner = Runner
    server.web.TCPSite = Site

    try:
        with tempfile.TemporaryDirectory() as temp:
            config = {
                'model_type': 'hunyuan_v1_dense',
                'hidden_size': 2048,
                'num_hidden_layers': 32,
                'vocab_size': 120818,
                'quantization_config': {'quant_method': 'exl3'},
            }
            (Path(temp) / 'config.json').write_text(json.dumps(config))
            opts = server.parse_args(['--model-dir', temp, '--max-concurrent-requests', '4'])
            opts.compute = 'unused'
            log = io.StringIO()
            try:
                with contextlib.redirect_stderr(log):
                    asyncio.run(server._serve(opts))
            except Ready:
                pass
            else:
                raise AssertionError('server failed to reach listener setup')
            assert opts.max_seq_len == 32768, opts.max_seq_len
    finally:
        server.codec.load_processor = orig_load
        server.codec.reasoning_capability = orig_probe
        server.AsyncComputeClient = orig_client
        server.ChatService = orig_service
        server.build_compute_argv = orig_argv
        server.create_app = orig_app
        server.web.AppRunner = orig_runner
        server.web.TCPSite = orig_site
    print('hunyuan_v1_dense startup skips reasoning probe cleanly')


if __name__ == '__main__':
    main()
