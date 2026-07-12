from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Protocol

from tileflow.language.ir import Operation


class Frame(Protocol):
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...


def unwrap_cond(expr):
    pass


class Builder:
    def __init__(self):
        self._region_frames: list[list[Operation]]
        self.frames: list[Frame] = []

    def enter_frame(self, frame: Frame):
        self.frames.append(frame)
        return frame.__enter__()

    @contextmanager
    def with_frame(self, frame: Frame):
        pop_idx = len(self.frames)
        yield self.enter_frame(frame)
        while len(self.frames) > pop_idx:
            self.frames.pop().__exit__(None, None, None)

    def ctx_if(self, cond):
        # expression?
        pass

    class _has_if_frame: ...


    def ctx_then(self, val):
        if val is self._has_if_frame:
            with self.with_frame()