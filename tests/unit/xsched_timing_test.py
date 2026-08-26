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

"""Tests for xsched synchronized contiguous-window timing."""

import ast
import json
from pathlib import Path

import pytest

from maxtext.common import xsched_timing


def _environment(**overrides: str) -> dict[str, str]:
  return {
      xsched_timing.PROTOCOL_ENV: xsched_timing.PROTOCOL,
      xsched_timing.START_STEP_ENV: "2",
      xsched_timing.STEP_COUNT_ENV: "3",
      **overrides,
  }


def test_spec_environment_is_all_or_none_and_strict() -> None:
  assert xsched_timing.SynchronizedWindowSpec.from_environment({}) is None
  assert xsched_timing.SynchronizedWindowSpec.from_environment(_environment()) == (
      xsched_timing.SynchronizedWindowSpec(start_step=2, step_count=3)
  )

  with pytest.raises(ValueError, match="incomplete"):
    xsched_timing.SynchronizedWindowSpec.from_environment(
        {xsched_timing.PROTOCOL_ENV: xsched_timing.PROTOCOL}
    )
  with pytest.raises(ValueError, match="unsupported"):
    xsched_timing.SynchronizedWindowSpec.from_environment(
        _environment(**{xsched_timing.PROTOCOL_ENV: "v2"})
    )
  with pytest.raises(ValueError, match="canonical"):
    xsched_timing.SynchronizedWindowSpec.from_environment(
        _environment(**{xsched_timing.START_STEP_ENV: "02"})
    )
  with pytest.raises(ValueError, match="positive"):
    xsched_timing.SynchronizedWindowSpec.from_environment(
        _environment(**{xsched_timing.START_STEP_ENV: "0"})
    )


def test_timer_synchronizes_only_at_boundaries_and_emits_one_marker() -> None:
  events: list[tuple[str, object]] = []
  clock_values = iter((100, 6100))
  functions = xsched_timing._TimingFunctions(  # pylint: disable=protected-access
      block_until_ready=lambda state: events.append(("block", state)),
      barrier=lambda name: events.append(("barrier", name)),
      clock_ns=lambda: events.append(("clock", None)) or next(clock_values),
      log=lambda message: events.append(("log", message)),
      rank=lambda: 7,
      rank_count=lambda: 128,
  )
  timer = xsched_timing.SynchronizedWindowTimer(
      xsched_timing.SynchronizedWindowSpec(start_step=2, step_count=3),
      first_step=0,
      final_step_exclusive=6,
      functions=functions,
  )

  timer.after_step(0, "state-0")
  assert events == []
  timer.after_step(1, "state-1")
  assert [event[0] for event in events] == ["block", "barrier", "clock"]
  timer.after_step(2, "state-2")
  timer.after_step(3, "state-3")
  assert [event[0] for event in events] == ["block", "barrier", "clock"]
  timer.after_step(4, "state-4")
  timer.require_complete()

  assert [event[0] for event in events] == [
      "block",
      "barrier",
      "clock",
      "block",
      "clock",
      "barrier",
      "log",
  ]
  marker_text = events[-1][1]
  assert isinstance(marker_text, str)
  assert marker_text.startswith(xsched_timing.MARKER_PREFIX)
  assert json.loads(marker_text.removeprefix(xsched_timing.MARKER_PREFIX)) == {
      "barrier_semantics": "boundary_barriers_excluded",
      "clock": "perf_counter_ns",
      "elapsed_ns": 6000,
      "end_step": 4,
      "protocol": xsched_timing.PROTOCOL,
      "rank": 7,
      "rank_count": 128,
      "schema_version": 1,
      "start_step": 2,
      "step_count": 3,
  }


def test_timer_requires_warmup_after_restored_step_and_complete_window() -> None:
  spec = xsched_timing.SynchronizedWindowSpec(start_step=100, step_count=2)
  with pytest.raises(ValueError, match="ordinary loop post-work"):
    xsched_timing.SynchronizedWindowTimer(
        spec, first_step=100, final_step_exclusive=103
    )

  spec = xsched_timing.SynchronizedWindowSpec(start_step=102, step_count=2)
  with pytest.raises(ValueError, match="training range"):
    xsched_timing.SynchronizedWindowTimer(
        spec, first_step=100, final_step_exclusive=103
    )


def test_timer_uses_absolute_steps_after_checkpoint_restore() -> None:
  events = []
  clock_values = iter((1_000, 5_001_000))
  functions = xsched_timing._TimingFunctions(  # pylint: disable=protected-access
      block_until_ready=lambda state: events.append(("block", state)),
      barrier=lambda name: events.append(("barrier", name)),
      clock_ns=lambda: next(clock_values),
      log=lambda message: events.append(("log", message)),
      rank=lambda: 0,
      rank_count=lambda: 1,
  )
  timer = xsched_timing.SynchronizedWindowTimer(
      xsched_timing.SynchronizedWindowSpec(start_step=102, step_count=2),
      first_step=100,
      final_step_exclusive=105,
      functions=functions,
  )
  for step in range(100, 104):
    timer.after_step(step, f"state-{step}")
  timer.require_complete()
  marker = json.loads(events[-1][1].removeprefix(xsched_timing.MARKER_PREFIX))
  assert (marker["start_step"], marker["end_step"], marker["step_count"]) == (
      102,
      103,
      2,
  )


def test_timer_fails_closed_on_incomplete_or_noncontiguous_execution() -> None:
  timer = xsched_timing.SynchronizedWindowTimer(
      xsched_timing.SynchronizedWindowSpec(start_step=2, step_count=2),
      first_step=0,
      final_step_exclusive=5,
  )
  with pytest.raises(RuntimeError, match="expected 0, got 1"):
    timer.after_step(1, object())
  with pytest.raises(RuntimeError, match="without emitting"):
    timer.require_complete()


def test_timed_stop_training_is_bare_rethrown_independent_of_window_phase() -> None:
  source = (
      Path(__file__).resolve().parents[2]
      / "src/maxtext/trainers/pre_train/train.py"
  ).read_text()
  handlers = [
      node
      for node in ast.walk(ast.parse(source))
      if isinstance(node, ast.ExceptHandler)
      and ast.unparse(node.type) == "exceptions.StopTraining"
  ]
  assert len(handlers) == 1
  timing_guards = [
      node
      for node in handlers[0].body
      if isinstance(node, ast.If)
      and ast.unparse(node.test) == "timing_window is not None"
  ]
  assert len(timing_guards) == 1
  assert len(timing_guards[0].body) == 1
  assert isinstance(timing_guards[0].body[0], ast.Raise)
  assert timing_guards[0].body[0].exc is None


def test_train_loop_boundary_is_after_ordinary_step_post_work() -> None:
  source = (
      Path(__file__).resolve().parents[2]
      / "src/maxtext/trainers/pre_train/train.py"
  ).read_text()
  metric_post_work = (
      "metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)"
  )
  timing_boundary = "timing_window.after_step(step, state)"
  assert source.count(timing_boundary) == 1
  assert source.index(metric_post_work) < source.index(timing_boundary)
