"""
Validate TrioSim trace files (tensor.csv + trace.csv) for internal consistency.

Usage:
    python tracer/validate_trace.py <trace_base_dir>

Where <trace_base_dir> contains one subdirectory per model, each holding
tensor.csv and trace.csv. Exits with code 1 if any errors are found.

Example:
    python tracer/validate_trace.py tracer/data/middledata/trace
    python tracer/validate_trace.py sample_trace/trace2-h100-bs128
"""

import os
import sys
import pandas as pd

_TENSOR_COLS = {'Index', 'TensorID', 'TensorShape', 'TensorNumElement',
                'TensorEachByte', 'TensorType', 'TensorStorgeid', 'gpuid'}
_TRACE_COLS = {'OperatorID', 'OperatorName', 'Operator_input', 'Operator_output',
               'Operator_cudatime', 'Operator_cudatimenooverlap',
               'InputSize', 'OutputSize', 'gpuid', 'stage', 'tpflag'}
_VALID_STAGES = {'forward', 'backward', 'optimizer'}
_VALID_TPFLAGS = {0, 1}


def _parse_id_list(raw):
    """Parse a semicolon-delimited ID list like '[4; 6; 8]' into a set of strings."""
    cleaned = str(raw).strip().strip('[]').replace(' ', '')
    if not cleaned:
        return set()
    return {tid for tid in cleaned.split(';') if tid}


def validate_trace_dir(trace_dir):
    """Return a list of error strings for the given model trace directory."""
    errors = []

    tensor_path = os.path.join(trace_dir, 'tensor.csv')
    trace_path = os.path.join(trace_dir, 'trace.csv')

    if not os.path.exists(tensor_path):
        return [f"Missing tensor.csv"]
    if not os.path.exists(trace_path):
        return [f"Missing trace.csv"]

    tensor_df = pd.read_csv(tensor_path)
    trace_df = pd.read_csv(trace_path)

    missing_tensor_cols = _TENSOR_COLS - set(tensor_df.columns)
    if missing_tensor_cols:
        errors.append(f"tensor.csv missing columns: {sorted(missing_tensor_cols)}")

    missing_trace_cols = _TRACE_COLS - set(trace_df.columns)
    if missing_trace_cols:
        errors.append(f"trace.csv missing columns: {sorted(missing_trace_cols)}")

    if errors:
        return errors

    if tensor_df.empty:
        errors.append("tensor.csv is empty")
    if trace_df.empty:
        errors.append("trace.csv is empty")
        return errors

    known_ids = set(tensor_df['TensorID'].astype(str))

    missing_refs = set()
    for _, row in trace_df.iterrows():
        for col in ('Operator_input', 'Operator_output'):
            for tid in _parse_id_list(row[col]):
                if tid not in known_ids:
                    missing_refs.add(tid)
    if missing_refs:
        sample = sorted(missing_refs)[:5]
        errors.append(
            f"{len(missing_refs)} tensor ID(s) referenced in trace.csv not found in tensor.csv "
            f"(sample: {sample})"
        )

    zero_time = (trace_df['Operator_cudatime'] == 0).sum()
    if zero_time:
        errors.append(f"{zero_time} operator(s) with zero Operator_cudatime")

    invalid_stages = trace_df[~trace_df['stage'].isin(_VALID_STAGES)]
    if not invalid_stages.empty:
        bad = invalid_stages['stage'].unique().tolist()
        errors.append(f"{len(invalid_stages)} operator(s) with invalid stage value(s): {bad}")

    invalid_flags = trace_df[~trace_df['tpflag'].isin(_VALID_TPFLAGS)]
    if not invalid_flags.empty:
        errors.append(f"{len(invalid_flags)} operator(s) with invalid tpflag (must be 0 or 1)")

    dup_ops = trace_df['OperatorID'].duplicated().sum()
    if dup_ops:
        errors.append(f"{dup_ops} duplicate OperatorID value(s) in trace.csv")

    return errors


def validate_all(trace_base_dir):
    """Validate all model subdirectories under trace_base_dir. Exits 1 on failure."""
    if not os.path.exists(trace_base_dir):
        print(f"ERROR: trace directory not found: {trace_base_dir}")
        sys.exit(1)

    entries = [
        e for e in os.listdir(trace_base_dir)
        if os.path.isdir(os.path.join(trace_base_dir, e))
    ]
    if not entries:
        print(f"WARNING: no model subdirectories found in {trace_base_dir}")
        return

    all_ok = True
    for model_name in sorted(entries):
        full_path = os.path.join(trace_base_dir, model_name)
        errors = validate_trace_dir(full_path)
        if errors:
            all_ok = False
            print(f"[FAIL] {model_name}")
            for e in errors:
                print(f"       - {e}")
        else:
            print(f"[OK]   {model_name}")

    if all_ok:
        print(f"\nAll {len(entries)} trace(s) valid.")
    else:
        print(f"\nValidation failed — see errors above.")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    validate_all(sys.argv[1])
