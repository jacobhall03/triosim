"""
Unified entry point for TrioSim trace generation.

Usage:
    python tracer/generate_trace.py [options]

Examples:
    # Trace vgg13 and resnet50 at batch size 128 (uses GPU)
    python tracer/generate_trace.py --models vgg13 resnet50 --batch-size 128

    # Process already-collected JSON data without re-running the profiler
    python tracer/generate_trace.py --skip-collection --models vgg13

    # Validate existing traces without re-generating
    python tracer/generate_trace.py --validate-only
"""

import argparse
import os
import sys
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

_DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, 'trace_config.yaml')

_HARDCODED_DEFAULTS = {
    'models': [
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
    ],
    'batch_sizes': [128],
    'num_iters': 43,
    'profile_start': 40,
    'warmup_start': 30,
    'dataset_path': os.path.join(_SCRIPT_DIR, 'ILSVRC2012_img_val'),
    'data_dir': os.path.join(_SCRIPT_DIR, 'data'),
    'workers': 8,
    'target_ops': ['aten::conv2d', 'aten::linear', 'aten::embedding'],
}


def _load_config(config_path):
    cfg = _HARDCODED_DEFAULTS.copy()
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
        for key in ('dataset_path', 'data_dir'):
            if key in file_cfg and not os.path.isabs(file_cfg[key]):
                cfg[key] = os.path.join(_SCRIPT_DIR, file_cfg[key])
    return cfg


def _apply_overrides(cfg, args):
    if args.models:
        cfg['models'] = args.models
    if args.batch_size is not None:
        cfg['batch_sizes'] = [args.batch_size]
    if args.target_ops:
        cfg['target_ops'] = args.target_ops
    if args.workers is not None:
        cfg['workers'] = args.workers
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Generate TrioSim traces from PyTorch models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', default=_DEFAULT_CONFIG_PATH,
                        help='Path to trace_config.yaml (default: tracer/trace_config.yaml)')
    parser.add_argument('--models', nargs='+', metavar='MODEL',
                        help='Model names to trace (overrides config). E.g. --models vgg13 resnet50')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='Batch size to use (overrides config batch_sizes list)')
    parser.add_argument('--target-ops', nargs='+', metavar='OP',
                        help='Operator names that set tpflag=1 (overrides config target_ops)')
    parser.add_argument('--workers', type=int, metavar='N',
                        help='DataLoader worker count (overrides config)')
    parser.add_argument('--method', choices=['original', 'fx'], default='original',
                        help=(
                            'Trace collection method. '
                            '"original": full-model profiler run (requires entire model in GPU memory). '
                            '"fx": torch.fx graph extraction + per-op microbenchmarks '
                            '(memory-agnostic; peak GPU usage = one layer at a time).'
                        ))
    parser.add_argument('--skip-collection', action='store_true',
                        help='[original only] Skip profiler data collection (use existing JSON files in data/)')
    parser.add_argument('--skip-processing', action='store_true',
                        help='[original only] Skip trace processing (use existing intermediate CSVs)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation of final trace files')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing traces, do not collect or process')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    trace_out_dir = os.path.join(cfg['data_dir'], 'middledata', 'trace')

    if args.validate_only:
        from validate_trace import validate_all
        validate_all(trace_out_dir)
        return

    if args.method == 'fx':
        print("=" * 60)
        print("FX Method: torch.fx graph extraction + per-op microbenchmarks")
        print("=" * 60)
        from datacollect_fx import collect_traces_fx
        collect_traces_fx(cfg)
    else:
        if not args.skip_collection:
            print("=" * 60)
            print("Step 1/2: Collecting traces from PyTorch profiler")
            print("=" * 60)
            from datacollect import collect_traces
            collect_traces(cfg)

        if not args.skip_processing:
            print("=" * 60)
            print("Step 2/2: Processing traces into TrioSim format")
            print("=" * 60)
            from dataprocess import process_traces
            process_traces(cfg)

    if not args.no_validate:
        print("=" * 60)
        print("Validating output traces")
        print("=" * 60)
        from validate_trace import validate_all
        validate_all(trace_out_dir)


if __name__ == '__main__':
    main()
