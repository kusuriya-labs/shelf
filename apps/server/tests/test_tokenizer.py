"""chat template の出力（token id 列）の正規化（T06 の入口）。

transformers 5 系の `apply_chat_template(tokenize=True)` は `BatchEncoding` を返す。
旧実装は `isinstance(out, dict)` で分岐していたため、`input_ids` を取り出せずに
キー文字列を int 変換して落ちていた。ここでは代表的な戻り値の形を固定する。
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import server as srv  # noqa: E402


class _BatchLike:
    """`BatchEncoding` の最小模倣（`keys` と `__getitem__` を持つ）。"""

    def __init__(self, data):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _TensorLike:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _EncodingLike:
    def __init__(self, ids):
        self.ids = ids


def test_batch_encoding():
    out = _BatchLike({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
    assert srv._ids_from_template_output(out) == [1, 2, 3]


def test_plain_dict():
    assert srv._ids_from_template_output({"input_ids": [4, 5]}) == [4, 5]


def test_nested_batch_dimension():
    assert srv._ids_from_template_output([[6, 7, 8]]) == [6, 7, 8]


def test_encoding_list():
    assert srv._ids_from_template_output([_EncodingLike([9, 10])]) == [9, 10]


def test_tensor_like():
    assert srv._ids_from_template_output(_TensorLike([[11, 12]])) == [11, 12]


def test_plain_list():
    assert srv._ids_from_template_output([13, 14]) == [13, 14]
