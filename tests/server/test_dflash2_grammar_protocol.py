import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/apps/server'))
from serverlib.compute_client import AsyncComputeClient, ComputeProtocolError


class Grammar:
    def __init__(self, delayed=False):
        self.after_reasoning = delayed
        self.accepted = []
        self.drafts = []

    def next_mask(self):
        return 'mask'

    def accept(self, token):
        self.accepted.append(token)
        return True

    def draft_masks(self, tokens):
        self.drafts.append(tokens)
        return min(1, len(tokens)), ['row'] * (min(1, len(tokens)) + 1)


class ProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, events):
        client = AsyncComputeClient([])
        mailbox = asyncio.Queue()
        for event in events:
            mailbox.put_nowait(event)
        sent = []
        client.ended = []
        async def begin():
            return 7, mailbox
        async def end(request_id):
            client.ended.append(request_id)
        async def send(payload):
            sent.append(payload)
        client._begin_request = begin
        client._end_request = end
        client._send = send
        return client, sent

    async def test_drafts_and_stale_rounds_both_modes(self):
        for streaming in (False, True):
            events = [{'event': 'token', 'token_id': 1, 'need_mask': False},
                      {'event': 'draft', 'round_id': 2, 'draft_tokens': [2, 3]},
                      {'event': 'draft', 'round_id': 1, 'draft_tokens': [9]},
                      {'event': 'token', 'token_id': 2, 'need_mask': False},
                      {'event': 'token', 'token_id': 4, 'need_mask': False},
                      {'event': 'done', 'finish_reason': 'length'}]
            client, sent = await self.make_client(events)
            grammar = Grammar()
            if streaming:
                result = [event async for event in client.generate_stream([1], 8, grammar=grammar)]
                self.assertEqual(result[-1]['done']['finish_reason'], 'length')
            else:
                result = await client.generate([1], 8, grammar=grammar)
                self.assertEqual(result['finish_reason'], 'length')
                self.assertEqual(client.ended, [7])
            self.assertEqual(grammar.accepted, [1, 2, 4])
            self.assertEqual(grammar.drafts, [[2, 3]])
            self.assertEqual(sent[1:], [{'op': 'draft_masks', 'request_id': 7, 'round_id': 2,
                                         'valid_prefix': 1, 'bitmasks': ['row', 'row']}])

    async def test_reasoning_boundary(self):
        events = [{'event': 'token', 'token_id': 5, 'need_mask': False},
                  {'event': 'token', 'token_id': 99, 'need_mask': False},
                  {'event': 'draft', 'round_id': 1, 'draft_tokens': []},
                  {'event': 'token', 'token_id': 8, 'need_mask': False},
                  {'event': 'done', 'finish_reason': 'length'}]
        client, sent = await self.make_client(events)
        grammar = Grammar(True)
        await client.generate([1], 8, reasoning_close_token=99, grammar=grammar)
        self.assertEqual(grammar.accepted, [8])
        self.assertEqual(grammar.drafts, [[]])
        self.assertEqual([x['op'] for x in sent], ['generate', 'draft_masks'])

    async def test_draft_before_reasoning_close_is_rejected(self):
        client, sent = await self.make_client([{'event': 'draft', 'round_id': 1, 'draft_tokens': []}])
        client._drain_terminal = lambda mailbox: asyncio.sleep(0)
        with self.assertRaises(ComputeProtocolError):
            await client.generate([1], 8, reasoning_close_token=99, grammar=Grammar(True))
        self.assertEqual([x['op'] for x in sent], ['generate', 'cancel'])


if __name__ == '__main__':
    unittest.main()
