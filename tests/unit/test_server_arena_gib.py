import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / "src/apps/server/shelf_server.py"
spec = importlib.util.spec_from_file_location("shelf_server", path)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ArenaGibTest(unittest.TestCase):
    def test_values(self):
        for value in ("0", "0.1", "0.3", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                server.arena_gib_value(value)
        for value, expected in (("auto", "auto"), ("0.31", 0.31), ("2", 2), ("-2", -2)):
            with self.subTest(value=value):
                self.assertEqual(server.arena_gib_value(value), expected)

    def test_parse_args(self):
        for value in ("0", "0.1", "0.3"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                server.parse_args(["--model-dir", "/model", "--arena-gib", value])
        for value, expected in (("auto", "auto"), ("0.31", 0.31), ("2", 2), ("-2", -2)):
            with self.subTest(value=value):
                self.assertEqual(server.parse_args(["--model-dir", "/model", "--arena-gib", value]).arena_gib, expected)
        self.assertEqual(server.parse_args(["--model-dir", "/model"]).arena_gib, "auto")


if __name__ == "__main__":
    unittest.main()
