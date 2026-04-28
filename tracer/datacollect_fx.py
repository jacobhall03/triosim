"""
Memory-agnostic trace collection using two passes:
  Pass 1 — torch.fx symbolic trace + ShapeProp on CPU
            → full operator graph and tensor shapes without loading the model on GPU
  Pass 2 — per-operator GPU microbenchmarks
            → cudatime for each op in isolation (peak GPU memory = one layer at a time)

Writes tensor.csv + trace.csv directly to data/middledata/trace/<model_name>/,
the same location and format expected by triosim/main.go via trace.go.
"""

import os
import torch
import torch.fx
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.fx.passes.shape_prop import ShapeProp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_DTYPE_BYTES = {
    torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    torch.float64: 8, torch.int32: 4, torch.int64: 8, torch.bool: 1,
}

# Map nn.Module class names to aten op names for tpflag detection
_MODULE_TO_ATEN = {
    'Conv1d': 'aten::conv1d',
    'Conv2d': 'aten::conv2d',
    'Conv3d': 'aten::conv3d',
    'Linear': 'aten::linear',
    'Embedding': 'aten::embedding',
}

DEFAULT_TARGET_OPS = ['aten::conv2d', 'aten::linear', 'aten::embedding']
DEFAULT_CONFIG = {
    'models': ['vgg13'],
    'batch_sizes': [32],
    'data_dir': os.path.join(_SCRIPT_DIR, 'data'),
    'target_ops': DEFAULT_TARGET_OPS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nbytes(dtype):
    return _DTYPE_BYTES.get(dtype, 4)


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _scale_shape(shape, bs):
    """Scale batch dimension (dim 0) to target batch size."""
    if not shape:
        return shape
    s = list(shape)
    s[0] = bs
    return s


def _get_meta(node):
    """Return (shape_list, dtype) from a node's ShapeProp tensor_meta, or (None, None)."""
    meta = node.meta.get('tensor_meta')
    if meta is None:
        return None, None
    if hasattr(meta, 'shape'):
        return list(meta.shape), getattr(meta, 'dtype', torch.float32)
    # Some nodes produce a tuple of tensors; use the first one.
    if isinstance(meta, (list, tuple)) and meta:
        m = meta[0]
        if hasattr(m, 'shape'):
            return list(m.shape), getattr(m, 'dtype', torch.float32)
    return None, None


def _make_tensor(shape, dtype, bs):
    """Create a random concrete tensor scaled to target batch size from a bs=1 shape."""
    if not shape:
        return None
    s = _scale_shape(shape, bs)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 100, s, dtype=dtype)
    safe = dtype if dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16) else torch.float32
    return torch.randn(s, dtype=safe)


# ---------------------------------------------------------------------------
# Tracing with fallback for models that have control flow (e.g. DenseNet)
# ---------------------------------------------------------------------------

class _AtomicTracer(torch.fx.Tracer):
    """
    Treat every module that is not a pure container as a leaf.
    This avoids tracing into modules whose forward() has data-dependent
    Python control flow (e.g. DenseNet's _DenseLayer isinstance check).
    Container modules (Sequential, ModuleList, ModuleDict) are still
    traced through so we get individual sub-module nodes.
    """
    def is_leaf_module(self, m, qualname):
        return not isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict))


def _symbolic_trace(model):
    """Try standard symbolic_trace; fall back to AtomicTracer on failure."""
    try:
        return torch.fx.symbolic_trace(model)
    except Exception as e_default:
        try:
            graph = _AtomicTracer().trace(model)
            return torch.fx.GraphModule(model, graph)
        except Exception as e_atomic:
            raise RuntimeError(
                f"torch.fx tracing failed.\n"
                f"  default tracer: {e_default}\n"
                f"  atomic tracer:  {e_atomic}"
            )


# ---------------------------------------------------------------------------
# Per-operator microbenchmark
# ---------------------------------------------------------------------------

def _benchmark(fn, gpu_inputs, warmup, iters):
    """
    Time fn(*gpu_inputs) on GPU; return (fwd_us, bwd_us) in microseconds.
    gpu_inputs must already be on the target device.
    bwd_us measures just the backward pass (not including forward).
    """
    float_inputs = [t for t in gpu_inputs
                    if isinstance(t, torch.Tensor) and t.is_floating_point()]
    for t in float_inputs:
        t.requires_grad_(True)

    # Warmup
    try:
        for _ in range(warmup):
            fn(*gpu_inputs)
        torch.cuda.synchronize()
    except Exception:
        return 0.0, 0.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Forward timing
    try:
        start.record()
        for _ in range(iters):
            fn(*gpu_inputs)
        end.record()
        torch.cuda.synchronize()
        fwd_us = start.elapsed_time(end) / iters * 1000  # ms → µs
    except Exception:
        return 0.0, 0.0

    # Backward timing: fresh graph each iter so we don't need retain_graph
    bwd_us = 0.0
    try:
        probe = fn(*gpu_inputs)
        if isinstance(probe, torch.Tensor) and probe.is_floating_point() and probe.requires_grad:
            total_ms = 0.0
            for _ in range(iters):
                fresh = [
                    t.detach().requires_grad_(t.is_floating_point())
                    if isinstance(t, torch.Tensor) else t
                    for t in gpu_inputs
                ]
                out = fn(*fresh)
                torch.cuda.synchronize()
                start.record()
                out.sum().backward()
                end.record()
                torch.cuda.synchronize()
                total_ms += start.elapsed_time(end)
                if hasattr(fn, 'zero_grad'):
                    fn.zero_grad(set_to_none=True)
            bwd_us = total_ms / iters * 1000
    except Exception:
        pass

    return fwd_us, bwd_us


def _benchmark_node(node, traced, bs, device, warmup, iters):
    """
    Build concrete GPU inputs for `node`, run the microbenchmark, clean up.
    Returns (fwd_us, bwd_us).
    """
    cpu_args = []
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            shape, dtype = _get_meta(arg)
            if shape is not None:
                t = _make_tensor(shape, dtype or torch.float32, bs)
                cpu_args.append(t if t is not None else torch.randn(1))
            else:
                cpu_args.append(None)
        elif isinstance(arg, (list, tuple)):
            # e.g. torch.cat takes a list of tensor nodes as its first arg
            inner = []
            for item in arg:
                if isinstance(item, torch.fx.Node):
                    shape, dtype = _get_meta(item)
                    if shape is not None:
                        t = _make_tensor(shape, dtype or torch.float32, bs)
                        if t is not None:
                            inner.append(t)
            cpu_args.append(type(arg)(inner) if inner else arg)
        else:
            # Scalar constants (stride, padding, dim, etc.) pass through as-is
            cpu_args.append(arg)

    # Drop trailing Nones (optional args that were absent)
    while cpu_args and cpu_args[-1] is None:
        cpu_args.pop()

    if not any(isinstance(a, torch.Tensor) for a in cpu_args):
        return 0.0, 0.0

    fwd_us, bwd_us = 0.0, 0.0
    try:
        if node.op == 'call_module':
            submod = traced.get_submodule(node.target).to(device).train()
            gpu_inputs = [a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_args]
            fwd_us, bwd_us = _benchmark(submod, gpu_inputs, warmup, iters)
            submod.cpu()
            del gpu_inputs

        elif node.op == 'call_function':
            gpu_inputs = [a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_args]
            fwd_us, bwd_us = _benchmark(node.target, gpu_inputs, warmup, iters)
            del gpu_inputs

        elif node.op == 'call_method':
            gpu_inputs = [a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_args]
            method = node.target

            def _method_call(*args, _m=method):
                return getattr(args[0], _m)(*args[1:])

            fwd_us, bwd_us = _benchmark(_method_call, gpu_inputs, warmup, iters)
            del gpu_inputs

    except Exception as e:
        print(f"    [warn] bench failed for {node.name} ({node.op} {node.target}): {e}")
    finally:
        torch.cuda.empty_cache()

    return fwd_us, bwd_us


# ---------------------------------------------------------------------------
# Input-arg collection for tensor ID tracking
# ---------------------------------------------------------------------------

def _collect_input_tids(node, node_tid, bs):
    """
    Walk node.args to collect (input_tids, input_sizes_bytes).
    Handles both flat tensor args and list-of-tensor args (e.g. torch.cat).
    """
    input_tids = []
    input_sizes = []

    def _add(arg_node):
        shape, dtype = _get_meta(arg_node)
        if shape is None:
            return
        tid = node_tid.get(arg_node.name)
        if tid is None:
            return
        scaled = _scale_shape(shape, bs)
        input_tids.append(tid)
        input_sizes.append(_numel(scaled))  # elements, not bytes (matches original format)

    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            _add(arg)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.fx.Node):
                    _add(item)

    return input_tids, input_sizes


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _run_one_model(model_name, bs, device, out_base, target_ops, warmup, iters):
    model_fn = getattr(models, model_name)
    model = model_fn(weights=None).eval()

    # --- Pass 1: graph extraction ---
    traced = _symbolic_trace(model)
    sp = ShapeProp(traced)
    sp.propagate(torch.randn(1, 3, 224, 224))

    # Assign a unique integer TensorID to every node that produces a tensor.
    node_tid = {}
    tid_counter = 0
    for node in traced.graph.nodes:
        shape, _ = _get_meta(node)
        if shape is not None:
            node_tid[node.name] = str(tid_counter)
            tid_counter += 1

    # Build tensor.csv rows (one row per unique tensor-producing node).
    tensor_rows = []
    for node in traced.graph.nodes:
        shape, dtype = _get_meta(node)
        if shape is None or node.name not in node_tid:
            continue
        tid = node_tid[node.name]
        if node.op == 'placeholder':
            ttype = 'input'
        elif node.op == 'get_attr':
            ttype = 'bias' if 'bias' in node.target else 'weight'
        else:
            ttype = 'output'
        scaled = _scale_shape(shape, bs)
        tensor_rows.append((
            tid, str(scaled), _numel(scaled),
            _nbytes(dtype or torch.float32), ttype, tid,
        ))

    # --- Pass 2: per-op microbenchmarks → build trace.csv rows ---
    op_rows = []
    op_idx = 0

    for node in traced.graph.nodes:
        if node.op not in ('call_module', 'call_function', 'call_method'):
            continue
        out_shape, out_dtype = _get_meta(node)
        if out_shape is None:
            continue

        input_tids, input_sizes = _collect_input_tids(node, node_tid, bs)
        out_tid = node_tid.get(node.name)
        scaled_out = _scale_shape(out_shape, bs)
        out_size = _numel(scaled_out)  # elements, not bytes (matches original format)

        fwd_us, bwd_us = 0.0, 0.0
        if torch.cuda.is_available():
            fwd_us, bwd_us = _benchmark_node(node, traced, bs, device, warmup, iters)

        # Determine human-readable op name and tpflag
        if node.op == 'call_module':
            submod = traced.get_submodule(node.target)
            mod_cls = type(submod).__name__
            op_name = mod_cls
            aten_name = _MODULE_TO_ATEN.get(mod_cls, '')
        elif node.op == 'call_function':
            op_name = getattr(node.target, '__name__', str(node.target))
            aten_name = ''
        else:
            op_name = str(node.target)
            aten_name = ''

        tpflag = 1 if (op_name in target_ops or aten_name in target_ops) else 0

        in_str = '[' + ';'.join(input_tids) + ']'
        out_str = '[' + (out_tid or '') + ']'
        in_sz_str = '[' + ';'.join(str(s) for s in input_sizes) + ']'
        out_sz_str = '[' + str(out_size) + ']'

        if fwd_us > 0:
            op_idx += 1
            op_rows.append((
                op_idx, op_name, in_str, out_str,
                fwd_us, fwd_us,
                in_sz_str, out_sz_str, '0', 'forward', tpflag,
            ))

        if bwd_us > 0:
            op_idx += 1
            op_rows.append((
                op_idx, f'autograd::{op_name}', out_str, in_str,
                bwd_us, bwd_us,
                out_sz_str, in_sz_str, '0', 'backward', 0,
            ))

    # --- Write output ---
    out_dir = os.path.join(out_base, model_name)
    os.makedirs(out_dir, exist_ok=True)

    tensor_df = pd.DataFrame(
        tensor_rows,
        columns=['TensorID', 'TensorShape', 'TensorNumElement',
                 'TensorEachByte', 'TensorType', 'TensorStorgeid'],
    )
    tensor_df.insert(0, 'Index', range(len(tensor_df)))
    tensor_df['gpuid'] = '0'
    tensor_df = tensor_df[['Index', 'TensorID', 'TensorShape', 'TensorNumElement',
                            'TensorEachByte', 'TensorType', 'TensorStorgeid', 'gpuid']]
    tensor_df.to_csv(os.path.join(out_dir, 'tensor.csv'), index=False)

    op_df = pd.DataFrame(
        op_rows,
        columns=['OperatorID', 'OperatorName', 'Operator_input', 'Operator_output',
                 'Operator_cudatime', 'Operator_cudatimenooverlap',
                 'InputSize', 'OutputSize', 'gpuid', 'stage', 'tpflag'],
    )
    op_df.to_csv(os.path.join(out_dir, 'trace.csv'), index=False)
    print(f"  wrote {len(tensor_df)} tensors, {len(op_df)} ops → {out_dir}")


def collect_traces_fx(config):
    """
    Public entry point — mirrors the signature of collect_traces() in datacollect.py.
    Called by generate_trace.py when --method fx is selected.
    """
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found — timings will be 0 (graph structure only).")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = config.get('data_dir', DEFAULT_CONFIG['data_dir'])
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_SCRIPT_DIR, data_dir)
    out_base = os.path.join(data_dir, 'middledata', 'trace')
    target_ops = config.get('target_ops', DEFAULT_TARGET_OPS)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Reuse num_iters / warmup_start from config as benchmark iters / warmup
    bench_iters = max(config.get('num_iters', 10), 3)
    bench_warmup = max(config.get('warmup_start', 5), 2)

    for bs in config['batch_sizes']:
        for model_name in config['models']:
            print(f"\n[FX] model={model_name}, batch_size={bs}")
            try:
                _run_one_model(model_name, bs, device, out_base,
                               target_ops, bench_warmup, bench_iters)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()


if __name__ == '__main__':
    import yaml, sys

    cfg = DEFAULT_CONFIG.copy()
    config_path = os.path.join(_SCRIPT_DIR, 'trace_config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
        for key in ('data_dir',):
            if key in file_cfg and not os.path.isabs(file_cfg[key]):
                cfg[key] = os.path.join(_SCRIPT_DIR, file_cfg[key])

    collect_traces_fx(cfg)
