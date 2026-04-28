#!/usr/bin/env bash
# Run the Hypothesis property-test lane.
#
# Why a script: invoking `pytest -m property tests/` collects all 549 tests
# from `tests/` just to filter to the 8 marked `property`. That triggers
# import-time pollution that hangs the run on Windows. Listing the files
# explicitly avoids the issue and is fast (~6s on a clean tree).
#
# Usage:
#   scripts/run_property_tests.sh            # fast profile (default)
#   HYPOTHESIS_PROFILE=ci scripts/run_property_tests.sh  # 4× more examples
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest \
  tests/test_animation_generator.py \
  tests/test_persistence_agent.py \
  tests/test_validation_agent.py \
  tests/test_json_sanitizer.py \
  tests/test_scene_controller.py \
  tests/test_determinism.py \
  -m property \
  -q "$@"
