import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ComputeHarness


def mask(token):
    data = bytearray(248320 // 8)
    data[token // 8] = 1 << (token % 8)
    return base64.b64encode(data).decode('ascii')


def main():
    if not os.environ.get('SHELF_DFLASH2_MODEL_DIR'):
        print('SKIP: SHELF_DFLASH2_MODEL_DIR is not set')
        return 77
    h = ComputeHarness(arena_gib=31, max_seq_len=256, max_concurrent_requests=2,
                       dflash2_model_dir=os.environ['SHELF_DFLASH2_MODEL_DIR'])
    try:
        h.start()
        h.wait_ready()
        for rid, temperature in ((10, 0.0), (11, 0.7)):
            expected = [100, 101, 102, 103]
            h.send({'op': 'generate', 'request_id': rid, 'input_ids': [248041, 77091],
                    'max_new_tokens': len(expected), 'temperature': temperature,
                    'grammar': True, 'bitmask': mask(expected[0])})
            seen = []
            while True:
                event = h.recv()
                assert event['event'] != 'error', event
                if event['event'] == 'token':
                    seen.append(event['token_id'])
                    assert not event.get('need_mask', True), event
                elif event['event'] == 'draft':
                    assert len(seen) < len(expected)
                    h.send({'op': 'draft_masks', 'request_id': rid, 'round_id': event['round_id'],
                            'valid_prefix': 0, 'bitmasks': [mask(expected[len(seen)])]})
                elif event['event'] == 'done':
                    assert seen == expected == event['generated_ids'], (event, seen)
                    break
        h.send({'op': 'generate', 'request_id': 13, 'input_ids': [248041, 77091],
                'max_new_tokens': 8, 'grammar': True, 'bitmask': mask(100)})
        seen = []
        proposed = None
        cancelled = False
        while True:
            event = h.recv()
            assert event['event'] != 'error', event
            if event['event'] == 'token':
                seen.append(event['token_id'])
                if len(seen) == 3:
                    h.send({'op': 'cancel', 'request_id': 13})
                    cancelled = True
            elif event['event'] == 'draft':
                if cancelled:
                    continue
                assert len(event['draft_tokens']) > 1, event
                proposed = event['draft_tokens'][0]
                h.send({'op': 'draft_masks', 'request_id': 13, 'round_id': event['round_id'],
                        'valid_prefix': 1, 'bitmasks': [mask(proposed), mask(101)]})
            elif event['event'] == 'done':
                assert cancelled and seen == [100, proposed, 101], seen
                assert event['finish_reason'] == 'cancelled', event
                break
        h.send({'op': 'generate', 'request_id': 4, 'input_ids': [248041, 77091],
                'max_new_tokens': 4, 'grammar': True, 'grammar_after_reasoning': True,
                'reasoning_budget_tokens': 1, 'reasoning_close_token': 104, 'bitmask': mask(100)})
        seen = []
        while True:
            event = h.recv()
            assert event['event'] != 'error', event
            if event['event'] == 'token':
                seen.append(event['token_id'])
            elif event['event'] == 'draft':
                assert len(seen) >= 2 and seen[1] == 104, seen
                h.send({'op': 'draft_masks', 'request_id': 4, 'round_id': event['round_id'],
                        'valid_prefix': 0, 'bitmasks': [mask(100 + len(seen) - 2)]})
            elif event['event'] == 'done':
                assert len(seen) == 4 and seen[1:] == [104, 100, 101], seen
                break
        h.send({'op': 'generate', 'request_id': 2, 'input_ids': [248041, 77091],
                'max_new_tokens': 10, 'grammar': True, 'bitmask': mask(100)})
        while True:
            event = h.recv()
            assert event['event'] != 'error', event
            if event['event'] == 'draft':
                old_round = event['round_id']
                break
        h.send_batch([{'op': 'draft_masks', 'request_id': 2, 'round_id': old_round + 1,
                       'valid_prefix': 0, 'bitmasks': [mask(101)]},
                      {'op': 'generate', 'request_id': 3, 'input_ids': [248041, 77091],
                       'max_new_tokens': 3}])
        while True:
            event = h.recv()
            assert event['event'] != 'error' and event.get('request_id') != 2, event
            if event.get('request_id') == 3 and event['event'] == 'done':
                break
        h.send_batch([{'op': 'cancel', 'request_id': 2},
                      {'op': 'draft_masks', 'request_id': 2, 'round_id': old_round,
                       'valid_prefix': 0, 'bitmasks': [mask(101)]}])
        event = h.recv()
        assert event['event'] == 'done' and event['finish_reason'] == 'cancelled', event
        for rid, invalid in ((20, 'prefix'), (21, 'length'), (22, 'round')):
            h.send({'op': 'generate', 'request_id': rid, 'input_ids': [248041, 77091],
                    'max_new_tokens': 10, 'grammar': True, 'bitmask': mask(100)})
            while True:
                event = h.recv()
                assert event['event'] != 'error', event
                if event['event'] == 'draft':
                    break
            h.send({'op': 'draft_masks', 'request_id': rid,
                    'round_id': -1 if invalid == 'round' else event['round_id'],
                    'valid_prefix': len(event['draft_tokens']) + 1 if invalid == 'prefix' else 0,
                    'bitmasks': [mask(101)] if invalid == 'prefix' else ['AAAA']})
            event = h.recv()
            assert event['event'] == 'error' and event['request_id'] == rid, event
        print('DFlash2 grammar protocol passed')
    finally:
        h.close()


if __name__ == '__main__':
    sys.exit(main())
