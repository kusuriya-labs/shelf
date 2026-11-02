"""Qwen3.8 generation region split.

A thinking-enabled generation starts inside the assistant turn's ``<think>``
block (the chat template writes the opening tag into the prompt) and leaves it
with a single ``</think>`` control token. This module turns that token sequence
into the two user-visible regions:

    generated token IDs
          -> REASONING region (tokens before ``</think>``)
          -> ANSWER region    (tokens after  ``</think>``)

The split is performed on raw token IDs, so no boundary text is ever decoded
into either region. Both the streaming and the non-streaming path drive the
same parser, which is what keeps ``join(stream deltas) == non_stream text``.

This module has no tokenizer dependency: ``decode`` is injected as a callable
so the state machine is testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

REASONING = "reasoning"
ANSWER = "answer"

REPLACEMENT_CHARACTER = "\ufffd"

__all__ = [
    "ANSWER",
    "REASONING",
    "GenerationRegionParser",
    "RegionDelta",
    "RegionText",
    "split_regions",
]


@dataclass(frozen=True)
class RegionDelta:
    region: str
    text: str


@dataclass(frozen=True)
class RegionText:
    reasoning: str
    answer: str
    closed: bool
    reasoning_tokens: int
    answer_tokens: int


class GenerationRegionParser:
    """Incremental ``REASONING`` / ``ANSWER`` splitter for one generation.

    ``close_ids`` is the token representation of the reasoning close tag as the
    active tokenizer encodes it. It may be a single control token or a short
    token sequence; the boundary is recognised only when the complete sequence
    has been generated, so a partial match is never emitted.

    ``thinking=False`` starts directly in ``ANSWER``: the non-thinking prompt
    already closes the thinking block, so nothing in the generation belongs to
    reasoning.
    """

    def __init__(self, decode: Callable[[Sequence[int]], str],
                 close_ids: Sequence[int], *, thinking: bool = True):
        self._decode = decode
        self._close_ids = tuple(int(token) for token in close_ids)
        self._thinking = bool(thinking)
        self._ids: list[int] = []
        self._scan = 0
        self._cut: Optional[int] = None
        self._reasoning = ""
        self._answer = ""
        self._finalized = False
        if not self._thinking:
            self._cut = 0

    @property
    def thinking(self) -> bool:
        return self._thinking

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def closed(self) -> bool:
        return self._cut is not None

    @property
    def token_count(self) -> int:
        return len(self._ids)

    @property
    def reasoning_tokens(self) -> int:
        return 0 if self._cut is None else self._reasoning_end()

    @property
    def answer_tokens(self) -> int:
        return 0 if self._cut is None else len(self._ids) - self._cut

    def text(self, region: str) -> str:
        return self._reasoning if region == REASONING else self._answer

    def emitted(self, region: str) -> str:
        return self.text(region)

    def resync(self, region: str, text: str) -> None:
        if region == REASONING:
            self._reasoning = text
        else:
            self._answer = text

    def feed(self, token_ids: Sequence[int]) -> list[RegionDelta]:
        if self._finalized:
            raise RuntimeError("generation region parser already finalized")
        for token_id in token_ids:
            self._ids.append(int(token_id))
        return self._advance(final=False)

    def finalize(self) -> list[RegionDelta]:
        if self._finalized:
            raise RuntimeError("generation region parser already finalized")
        self._finalized = True
        return self._advance(final=True)

    def result(self) -> RegionText:
        return RegionText(
            reasoning=self._reasoning,
            answer=self._answer,
            closed=self.closed,
            reasoning_tokens=self.reasoning_tokens,
            answer_tokens=self.answer_tokens,
        )

    def _advance(self, *, final: bool) -> list[RegionDelta]:
        self._locate_boundary()
        deltas: list[RegionDelta] = []
        if self._cut is None:
            self._emit(REASONING, self._decode(self._ids), deltas, final=final)
            return deltas
        self._emit(REASONING, self._decode(self._ids[:self._reasoning_end()]),
                   deltas, final=final)
        self._emit(ANSWER, self._decode(self._ids[self._cut:]),
                   deltas, final=final)
        return deltas

    def _reasoning_end(self) -> int:
        if self._cut is None:
            return 0
        return max(0, self._cut - len(self._close_ids))

    def _locate_boundary(self) -> None:
        if self._cut is not None:
            return
        width = len(self._close_ids)
        if width == 0 or len(self._ids) < width:
            return
        limit = len(self._ids) - width
        position = self._scan
        while position <= limit:
            if tuple(self._ids[position:position + width]) == self._close_ids:
                self._cut = position + width
                return
            position += 1
        self._scan = max(self._scan, limit)

    def _emit(self, region: str, full: str, deltas: list[RegionDelta],
              *, final: bool) -> None:
        if region == ANSWER:
            full = full.lstrip()
            current = self._answer
        else:
            current = self._reasoning

        if full == current:
            return
        if not final and full.endswith(REPLACEMENT_CHARACTER) \
                and full.startswith(current):
            return
        if not full.startswith(current):
            if not final:
                return
            delta = ""
        else:
            delta = full[len(current):]
        if region == ANSWER:
            self._answer = full
        else:
            self._reasoning = full
        if delta:
            deltas.append(RegionDelta(region=region, text=delta))


def split_regions(decode: Callable[[Sequence[int]], str],
                  generated_ids: Sequence[int], close_ids: Sequence[int],
                  *, thinking: bool = True) -> RegionText:
    parser = GenerationRegionParser(decode, close_ids, thinking=thinking)
    parser.feed(generated_ids)
    parser.finalize()
    return parser.result()
