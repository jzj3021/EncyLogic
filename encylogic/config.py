"""Central configuration for EncyLogic.

Every path, binary, and service endpoint the pipeline needs is resolved here from
environment variables, with sensible defaults. No script should hard-code an
absolute path -- import from this module instead.

Override any value via environment variables (see ``.env.example``), e.g.::

    export ENCYLOGIC_ROOT=/path/to/workspace
    export ABC_BIN=/opt/abc/abc
    export ENCYLOGIC_LIB=/path/to/asap7_merged.lib
    export ENCYLOGIC_API_URL=http://127.0.0.1:8088/evaluate
"""
from __future__ import annotations
import os
from pathlib import Path

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

ROOT = Path(_env("ENCYLOGIC_ROOT", str(Path(__file__).resolve().parents[2] / "workspace")))

ABC_BIN = _env("ABC_BIN", "abc")
YOSYS_BIN = _env("YOSYS_BIN", "yosys")
OPENROAD_BIN = _env("OPENROAD_BIN", "openroad")

LIB_PATH = _env("ENCYLOGIC_LIB", str(ROOT / "lib" / "asap7_merged_RVT_TT_NOSRAM.lib"))

API_URL = _env("ENCYLOGIC_API_URL", "http://127.0.0.1:8088/evaluate")

BENCHMARK_DIR = Path(_env("ENCYLOGIC_BENCHMARKS", str(ROOT / "benchmarks")))
VERILOG_DIR = ROOT / "verilog"
BLIF_DIR = ROOT / "blif"
AIG_DIR = ROOT / "aig"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = Path(_env("ENCYLOGIC_MODELS", str(ROOT / "models")))

SFT_CHECKPOINT = MODEL_DIR / "seqformer_sft.pth"
GRPO_CHECKPOINT = MODEL_DIR / "seqformer_grpo.pth"

PKG_DIR = Path(__file__).resolve().parent
ABC_RC = Path(_env("ENCYLOGIC_ABC_RC", str(PKG_DIR.parents[1] / "configs" / "abc.rc")))

ORFS_FLOW = Path(_env("ORFS_FLOW", "/opt/OpenROAD-flow-scripts/flow"))
DESIGNS_DIR = Path(_env("ENCYLOGIC_DESIGNS", str(ROOT / "designs")))
YOSYS_ABC_BIN = _env("YOSYS_ABC_BIN", "yosys-abc")
PLATFORM = _env("ENCYLOGIC_PLATFORM", "asap7")
SERVER_HOST = _env("ENCYLOGIC_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(_env("ENCYLOGIC_SERVER_PORT", "8088"))

def ensure_dirs() -> None:
    """Create the workspace directory tree if missing (safe to call repeatedly)."""
    for d in (BENCHMARK_DIR, VERILOG_DIR, BLIF_DIR, AIG_DIR, DATA_DIR, RESULTS_DIR, MODEL_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
