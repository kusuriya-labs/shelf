from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile

original = subprocess.run


def locked_run(cmd, *args, **kwargs):
    if isinstance(cmd, list) and cmd and Path(cmd[0]).name == 'shelf-compute':
        cmd = ['flock', '/tmp/r9700.lock', 'timeout', '-k', '5', '270', *cmd]
    return original(cmd, *args, **kwargs)


@contextmanager
def preserved_temporary_directory():
    yield tempfile.mkdtemp(prefix='tree-logits-', dir='/tmp/fc_dflash2graph')


subprocess.run = locked_run
path = Path('tests/server/test_compute_dflash2_tree.py').resolve()
source = path.read_text()
old = 'with tempfile.TemporaryDirectory() as tmp:'
if source.count(old) != 1:
    raise RuntimeError('temporary logits directory not found')
source = source.replace(old, 'with preserved_temporary_directory() as tmp:')
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path),
                                        'preserved_temporary_directory': preserved_temporary_directory})
