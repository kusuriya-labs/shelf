#!/usr/bin/env python3
"""CPU checks for speculative grammar masks and rollback."""
import base64
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/apps/server'), str(ROOT / 'src/apps/common')]
import xgrammar as xgr
from serverlib.structured_output import GrammarState


def allows(mask, token):
    data = base64.b64decode(mask)
    return bool(data[token // 8] & (1 << (token % 8)))


class SpeculativeGrammarTest(unittest.TestCase):
    def setUp(self):
        # Sequence "ab" followed by EOS; x is always invalid.
        info = xgr.TokenizerInfo(['a', 'b', 'x', '<eos>'], stop_token_ids=[3])
        compiled = xgr.GrammarCompiler(info).compile_regex('ab')
        self.state = GrammarState(xgr, compiled, 4)

    def test_valid_prefix_and_rollback(self):
        before = self.state.next_mask()
        count, rows = self.state.draft_masks([0, 1, 2])
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 3)
        self.assertTrue(allows(rows[0], 0))
        self.assertTrue(allows(rows[1], 1))
        self.assertTrue(allows(rows[2], 3))
        self.assertFalse(allows(rows[2], 2))
        self.assertEqual(self.state.next_mask(), before)
        self.assertTrue(self.state.accept(0))
        self.assertEqual(self.state.next_mask(), rows[1])

    def test_termination_and_empty_draft(self):
        count, rows = self.state.draft_masks([0, 1, 3, 2])
        self.assertEqual((count, len(rows)), (2, 3))
        self.assertTrue(allows(rows[-1], 3))
        self.assertEqual(self.state.draft_masks([]), (0, [rows[0]]))

    def test_invalid_first_token(self):
        count, rows = self.state.draft_masks([2, 0])
        self.assertEqual((count, len(rows)), (0, 1))
        self.assertTrue(allows(rows[0], 0))

    def test_rollback_on_mask_failure(self):
        before = self.state.next_mask()
        with patch.object(self.state, 'next_mask', side_effect=[before, RuntimeError('mask failure')]):
            with self.assertRaisesRegex(RuntimeError, 'mask failure'):
                self.state.draft_masks([0])
        self.assertEqual(self.state.next_mask(), before)


if __name__ == '__main__':
    unittest.main()
