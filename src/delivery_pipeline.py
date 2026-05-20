import sys

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("delivery_pipeline")


EPSILON: Final[float] = 1e-5
MIN_OSRM_TIME: Final[float] = 0.1
MAX_DELAY_RATIO: Final[float] = 20.0
MIN_DELAY_RATIO: Final[float] = 0.1

DELAY_MILD: Final[float] = 1.25
DELAY_MODERATE: Final[float] = 1.75
DELAY_SEVERE: Final[float] = 2.50

DATETIME_COLS: Final[list[str]] = [
    "trip_creation_time",
    "od_start_time",
    "od_end_time",
    "cutoff_timestamp",
]
