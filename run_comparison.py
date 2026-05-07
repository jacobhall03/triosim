#!/usr/bin/env python3
"""
Run triosim for both original and FX traces and collect predicted runtimes.

Expects the triosim binary at $PROJECT_DIR/triosim_bin (build with
  go build -o triosim_bin ./triosim/
from the project root before running this script).

Outputs: comparison_results.csv in the project root.
"""

import csv
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIOSIM_BIN = os.path.join(PROJECT_DIR, "triosim_bin")

BATCH_SIZE = 32
NUM_GPUS   = 4
CAPACITY   = 40          # default; each GPU gets 1 << 40 bytes

MODELS = ["vgg13", "densenet161", "resnet50"]

# case number → human-readable name (matches triosim --case flag)
PARALLELISM_CASES = {
    "inference":                 0,
    "data_parallel":             1,
    "distributed_data_parallel": 2,
    "tensor_parallel":           3,
    "pipeline_parallel":         4,
}

# interconnect value → name (matches triosim --interconnects flag)
INTERCONNECT_TYPES = {
    "electrical": 0,
    "optical":    1,
}

# For pipeline parallel: split the batch evenly across GPUs.
MICRO_BATCH_SIZE = BATCH_SIZE // NUM_GPUS   # 8

# Trace root directories (relative to project root, under tracer/).
_TRACER_DIR = os.path.join(PROJECT_DIR, "tracer")
ORIG_TRACE_ROOT = os.path.join(_TRACER_DIR, "data_compare_orig", "middledata", "trace")
FX_TRACE_ROOT   = os.path.join(_TRACER_DIR, "data_compare_fx",   "middledata", "trace")

OUTPUT_CSV = os.path.join(PROJECT_DIR, "comparison_results.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_dir(method: str, model: str) -> str:
    if method == "orig":
        return os.path.join(ORIG_TRACE_ROOT, model)
    # FX traces land under bs{N}/{model}/ after the batch-size directory fix.
    return os.path.join(FX_TRACE_ROOT, f"bs{BATCH_SIZE}", model)


def _extract_sim_time_ms(stdout: str):
    """
    Return the last simulated time (in ms) printed by triosim.

    triosim prints lines of the form:
        Estimated execution time ms, 12345.6789012345
        Current time after <...> ms, 12345.6789012345
    The 'Program Execution time: Xs' wall-clock line does NOT match because
    it has no comma after 'ms'.
    """
    matches = re.findall(r'ms,\s*([\d.]+)', stdout)
    if not matches:
        return None
    return float(matches[-1])


def _run_triosim(trace_dir: str, case: int, interconnect: int) -> float | None:
    """
    Invoke the triosim binary and return the simulated runtime in ms,
    or None on failure.
    """
    cmd = [
        TRIOSIM_BIN,
        "--trace-dir",      trace_dir,
        "--batch-size",     str(BATCH_SIZE),
        "--batch-size-sim", str(BATCH_SIZE),
        "--GPUnumber",      str(NUM_GPUS),
        "--capacity",       str(CAPACITY),
        "--case",           str(case),
        "--interconnects",  str(interconnect),
    ]

    if interconnect == 1:   # optical: numRows * numCols must equal GPUNumber
        cmd += ["--numRows", "1", "--numCols", str(NUM_GPUS)]

    if case == 4:           # pipeline parallel
        cmd += ["--micro-batch-size", str(MICRO_BATCH_SIZE)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_DIR,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {' '.join(cmd[-6:])}")
        return None

    if result.returncode != 0:
        print(f"    FAILED (rc={result.returncode}): {result.stderr.strip()[:200]}")
        return None

    t = _extract_sim_time_ms(result.stdout)
    if t is None:
        print(f"    WARNING: could not parse time from stdout:\n{result.stdout[:300]}")
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(TRIOSIM_BIN):
        print(f"ERROR: triosim binary not found at {TRIOSIM_BIN}")
        print("Build it first:  go build -o triosim_bin ./triosim/")
        sys.exit(1)

    fieldnames = [
        "model",
        "batch_size",
        "num_gpus",
        "capacity",
        "interconnect_type",
        "parallelism_type",
        "micro_batch_size",
        "non_fx_predicted_runtime_ms",
        "fx_predicted_runtime_ms",
    ]

    rows = []

    for model in MODELS:
        orig_dir = _trace_dir("orig", model)
        fx_dir   = _trace_dir("fx",   model)

        orig_exists = os.path.isdir(orig_dir)
        fx_exists   = os.path.isdir(fx_dir)

        if not orig_exists:
            print(f"WARNING: orig trace missing for {model}: {orig_dir}")
        if not fx_exists:
            print(f"WARNING: fx trace missing for {model}: {fx_dir}")

        for parallelism, case in PARALLELISM_CASES.items():
            for interconnect_name, interconnect in INTERCONNECT_TYPES.items():
                print(
                    f"  {model:12s}  {parallelism:30s}  {interconnect_name:10s}",
                    end="  ", flush=True,
                )

                orig_time = (
                    _run_triosim(orig_dir, case, interconnect)
                    if orig_exists else None
                )
                fx_time = (
                    _run_triosim(fx_dir, case, interconnect)
                    if fx_exists else None
                )

                print(f"orig={orig_time}  fx={fx_time}")

                rows.append({
                    "model":                       model,
                    "batch_size":                  BATCH_SIZE,
                    "num_gpus":                    NUM_GPUS,
                    "capacity":                    CAPACITY,
                    "interconnect_type":           interconnect_name,
                    "parallelism_type":            parallelism,
                    "micro_batch_size":            MICRO_BATCH_SIZE if case == 4 else "",
                    "non_fx_predicted_runtime_ms": orig_time,
                    "fx_predicted_runtime_ms":     fx_time,
                })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
