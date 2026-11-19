#!/usr/bin/env python3
"""Primary model-unit classification matched to the human-contact rule."""

import sys
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from pipeline_code.step03_classify_units.model_unit_classification_core import run_cli


if __name__ == "__main__":
    run_cli(
        method="human_nominal_consecutive2",
        default_output_name="step03_classify_units_human_nominal_consecutive2",
        description=(
            "Primary classification: positive factor-controlled effect and nominal "
            "one-sided permutation p<=.05 in at least two adjacent native windows."
        ),
    )
