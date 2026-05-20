import sys

"""
delivery_pipeline.py
====================
Production-grade preprocessing pipeline for logistics/delivery trip data.

Targets
-------
  • Graph Neural Networks   – node/edge feature tensors, adjacency ready
  • GRU/LSTM forecasting    – temporal sequences per corridor
  • Delay prediction        – robust target labels + engineered features

Author  : Senior Data / ML Engineering
Version : 2.0.0
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("delivery_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (single source of truth – never scatter magic numbers)
# ─────────────────────────────────────────────────────────────────────────────

EPSILON: Final[float] = 1e-5          # avoid division by zero
MIN_OSRM_TIME: Final[float] = 0.1    # minutes – discard degenerate rows
MAX_DELAY_RATIO: Final[float] = 20.0 # cap extreme outliers for model stability
MIN_DELAY_RATIO: Final[float] = 0.1  # floor (segment faster than OSRM floor)

# Delay-severity thresholds (minutes, relative to OSRM)
DELAY_MILD: Final[float]     = 1.25
DELAY_MODERATE: Final[float] = 1.75
DELAY_SEVERE: Final[float]   = 2.50

DATETIME_COLS: Final[list[str]] = [
    "trip_creation_time",
    "od_start_time",
    "od_end_time",
    "cutoff_timestamp",
]