# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Xsched synchronized contiguous-window timing."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import time
from typing import Any

import jax
from jax.experimental import multihost_utils

from maxtext.utils import max_logging


PROTOCOL = "xsched-synchronized-window-v1"
MARKER_PREFIX = "XSCHED_TIMING_WINDOW "
PROTOCOL_ENV = "XSCHED_TIMING_WINDOW_PROTOCOL"
START_STEP_ENV = "XSCHED_TIMING_WINDOW_START_STEP"
STEP_COUNT_ENV = "XSCHED_TIMING_WINDOW_STEP_COUNT"
ENVIRONMENT_VARIABLES = (PROTOCOL_ENV, START_STEP_ENV, STEP_COUNT_ENV)


@dataclass(frozen=True)
class SynchronizedWindowSpec:
  """Absolute training-step interval measured by xsched."""

  start_step: int
  step_count: int

  def __post_init__(self) -> None:
    if type(self.start_step) is not int or self.start_step <= 0:
      raise ValueError("xsched timing-window start step must be positive")
    if type(self.step_count) is not int or self.step_count <= 0:
      raise ValueError("xsched timing-window step count must be positive")

  @property
  def end_step(self) -> int:
    return self.start_step + self.step_count - 1

  @classmethod
  def from_environment(cls, environment: Mapping[str, str]) -> "SynchronizedWindowSpec | None":
    present = {name for name in ENVIRONMENT_VARIABLES if name in environment}
    if not present:
      return None
    if present != set(ENVIRONMENT_VARIABLES):
      missing = sorted(set(ENVIRONMENT_VARIABLES) - present)
      raise ValueError(f"incomplete xsched timing-window environment; missing {missing}")
    if environment[PROTOCOL_ENV] != PROTOCOL:
      raise ValueError(f"unsupported xsched timing-window protocol: {environment[PROTOCOL_ENV]!r}")
    return cls(
        start_step=_parse_canonical_integer(START_STEP_ENV, environment[START_STEP_ENV]),
        step_count=_parse_canonical_integer(STEP_COUNT_ENV, environment[STEP_COUNT_ENV]),
    )


@dataclass(frozen=True)
class _TimingFunctions:
  block_until_ready: Callable[[Any], Any]
  barrier: Callable[[str], None]
  clock_ns: Callable[[], int]
  log: Callable[[str], None]
  rank: Callable[[], int]
  rank_count: Callable[[], int]


_DEFAULT_FUNCTIONS = _TimingFunctions(
    block_until_ready=jax.block_until_ready,
    barrier=multihost_utils.sync_global_devices,
    clock_ns=time.perf_counter_ns,
    log=max_logging.log,
    rank=jax.process_index,
    rank_count=jax.process_count,
)


class SynchronizedWindowTimer:
  """Measures one contiguous interval without synchronizing its training steps."""

  def __init__(
      self,
      spec: SynchronizedWindowSpec,
      *,
      first_step: int,
      final_step_exclusive: int,
      functions: _TimingFunctions = _DEFAULT_FUNCTIONS,
  ):
    if spec.start_step <= first_step:
      raise ValueError(
          "xsched timing window must start after at least one completed ordinary "
          "loop post-work step; "
          f"first_step={first_step}, start_step={spec.start_step}"
      )
    if spec.end_step >= final_step_exclusive:
      raise ValueError(
          "xsched timing window exceeds the configured training range; "
          f"end_step={spec.end_step}, final_step_exclusive={final_step_exclusive}"
      )
    self._spec = spec
    self._functions = functions
    self._next_step = first_step
    self._start_ns: int | None = None
    self._emitted = False

  def after_step(self, step: int, state: Any) -> None:
    """Records the ordinary loop post-work boundary for `step`."""
    step = int(step)
    if step != self._next_step:
      raise RuntimeError(f"noncontiguous training steps: expected {self._next_step}, got {step}")
    self._next_step += 1

    if step == self._spec.start_step - 1:
      self._functions.block_until_ready(state)
      self._functions.barrier(self._barrier_name("start"))
      self._start_ns = self._functions.clock_ns()
      return
    if step != self._spec.end_step:
      return
    if self._start_ns is None:
      raise RuntimeError("xsched timing window ended before its start boundary")

    self._functions.block_until_ready(state)
    end_ns = self._functions.clock_ns()
    self._functions.barrier(self._barrier_name("end"))
    elapsed_ns = end_ns - self._start_ns
    if elapsed_ns <= 0:
      raise RuntimeError(f"xsched timing window has nonpositive elapsed time: {elapsed_ns}")
    if self._emitted:
      raise RuntimeError("xsched timing-window marker was already emitted")
    self._functions.log(MARKER_PREFIX + self._marker_json(elapsed_ns))
    self._emitted = True

  def require_complete(self) -> None:
    if not self._emitted:
      raise RuntimeError(
          "training completed without emitting the configured xsched timing-window marker; "
          f"window={self._spec.start_step}..{self._spec.end_step}"
      )

  def _barrier_name(self, boundary: str) -> str:
    return f"xsched-timing-{boundary}-{self._spec.start_step}-{self._spec.end_step}"

  def _marker_json(self, elapsed_ns: int) -> str:
    marker = {
        "barrier_semantics": "boundary_barriers_excluded",
        "clock": "perf_counter_ns",
        "elapsed_ns": elapsed_ns,
        "end_step": self._spec.end_step,
        "protocol": PROTOCOL,
        "rank": self._functions.rank(),
        "rank_count": self._functions.rank_count(),
        "schema_version": 1,
        "start_step": self._spec.start_step,
        "step_count": self._spec.step_count,
    }
    return json.dumps(marker, sort_keys=True, separators=(",", ":"))


def _parse_canonical_integer(name: str, value: str) -> int:
  if not value.isdecimal() or str(int(value)) != value:
    raise ValueError(f"{name} must be a canonical nonnegative integer, got {value!r}")
  return int(value)
