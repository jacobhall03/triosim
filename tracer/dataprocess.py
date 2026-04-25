import json
import sys
import os
import re
import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TARGET_OPS = [
    'aten::conv2d',
    'aten::linear',
    'aten::embedding',
]

DEFAULT_CONFIG = {
    'data_dir': os.path.join(_SCRIPT_DIR, 'data'),
    'target_ops': DEFAULT_TARGET_OPS,
}

_PROFILER_COLS = ['modelid', 'layerid', 'cpueventname', 'cudatime', 'cudatimenooverlap', 'inputdims', 'sequenceid']
_GRAPH_COLS = ['modelid', 'layerid', 'cpueventname', 'inputshapes', 'inputvalues', 'inputtypes',
               'outputshapes', 'outputvalues', 'outputtypes', 'op_schema']
_MERGE_COLS = ['layerid', 'cpueventname', 'inputshapes', 'inputvalues', 'inputtypes',
               'outputshapes', 'outputvalues', 'outputtypes', 'op_schema', 'cudatime',
               'cudatimenooverlap', 'stage', 'tpflag']


def _validate_df(df, required_cols, stage_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{stage_name}] Output is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"[{stage_name}] Produced an empty dataframe — check input files")


class _Kernel:
    __slots__ = ['name', 'duration', 'starttime', 'endtime', 'correlationid',
                 'registersperthread', 'sharedmemory', 'warpsperSM', 'grid', 'block', 'stream']

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.name = ''
        self.grid = ''
        self.block = ''


class _Operator:
    __slots__ = ['name', 'duration', 'starttime', 'endtime', 'correlationid', 'cudaevents',
                 'inputdims', 'sequenceid']

    def __init__(self):
        self.name = ''
        self.duration = None
        self.starttime = None
        self.endtime = None
        self.correlationid = []
        self.cudaevents = []
        self.inputdims = []
        self.sequenceid = None


def _process_layer(config):
    """Parse profiler JSON files → intermediate CSV with per-layer CUDA timing."""
    src_dir = os.path.join(config['data_dir'], 'profiler')
    out_dir = os.path.join(config['data_dir'], 'middledata', 'profiler')
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if fname == '.DS_Store':
            continue
        print(f"  [layer] {fname}")
        with open(os.path.join(src_dir, fname)) as f:
            json_trace = json.load(f)

        cpu_events = []
        profiler_start = 0

        for event in json_trace['traceEvents']:
            cat = event.get('cat', '').lower()
            ph = event.get('ph', '').lower()
            name = event.get('name', '')

            if (cat == 'cpu_op' or cat == 'operator') and ph == 'x':
                dur, ts = event['dur'], event['ts']
                te = ts + dur
                op = _Operator()
                op.name = name
                op.duration = dur
                op.starttime = ts
                op.endtime = te
                op.inputdims = event['args'].get('Input Dims', [])
                op.sequenceid = event['args'].get('Sequence number')

                to_remove = []
                for existing in cpu_events:
                    if (te <= existing.endtime and ts > existing.starttime) or \
                       (te < existing.endtime and ts >= existing.starttime) or \
                       (te == existing.endtime and ts == existing.starttime and op.name != existing.name):
                        to_remove.append(op)
                    elif te >= existing.endtime and ts < existing.starttime:
                        to_remove.append(existing)
                for item in to_remove:
                    if item in cpu_events:
                        cpu_events.remove(item)
                cpu_events.append(op)

            elif (cat == 'cuda_runtime' or cat == 'runtime') and ph == 'x' and \
                    name.lower() == 'cudalaunchkernel':
                corr = event['args']['correlation']
                ts, te = event['ts'], event['ts'] + event['dur']
                for op in cpu_events:
                    if op.endtime > te and op.starttime < ts:
                        op.correlationid.append(corr)

            elif name == 'Iteration Start: PyTorch Profiler':
                profiler_start = event.get('ts', 0)

        for event in json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel' and event.get('ph', '').lower() == 'x':
                corr = event['args']['correlation']
                dur, ts = event['dur'], event['ts']
                te = ts + dur
                for op in cpu_events:
                    if corr in op.correlationid:
                        k = _Kernel()
                        k.name = event['name']
                        k.duration = dur
                        k.starttime = ts
                        k.endtime = te
                        k.correlationid = corr
                        k.registersperthread = event['args']['registers per thread']
                        k.sharedmemory = event['args']['shared memory']
                        k.warpsperSM = event['args']['warps per SM']
                        k.grid = event['args']['grid']
                        k.block = event['args']['block']
                        k.stream = event['args']['stream']
                        op.cudaevents.append([k])

        cpu_events.sort(key=lambda x: x.starttime)
        rows = []
        for layer_id, op in enumerate(cpu_events, start=1):
            min_cuda, max_cuda, no_overlap = 0, 0, 0
            for kernel_group in op.cudaevents:
                for k in kernel_group:
                    t_start = k.starttime - profiler_start
                    t_end = k.endtime - profiler_start
                    if min_cuda == 0 or t_start < min_cuda:
                        min_cuda = t_start
                    if t_end > max_cuda:
                        max_cuda = t_end
                    no_overlap += k.endtime - k.starttime
            rows.append({
                'modelid': fname.replace('.json', '').replace('profiler_', ''),
                'layerid': layer_id,
                'cpueventname': op.name,
                'cudatime': max_cuda - min_cuda,
                'cudatimenooverlap': no_overlap,
                'inputdims': json.dumps(op.inputdims),
                'sequenceid': op.sequenceid,
            })

        df = pd.DataFrame(rows)
        _validate_df(df, _PROFILER_COLS, f'_process_layer/{fname}')
        out_path = os.path.join(out_dir, fname.replace('.json', '.csv'))
        df.to_csv(out_path, index=False)


def _process_graph(config):
    """Parse execution graph JSON files → intermediate CSV with tensor metadata."""
    src_dir = os.path.join(config['data_dir'], 'graph')
    out_dir = os.path.join(config['data_dir'], 'middledata', 'graph')
    os.makedirs(out_dir, exist_ok=True)

    for file_id, fname in enumerate(os.listdir(src_dir)):
        if fname == '.DS_Store':
            continue
        if file_id >= 100:
            break
        print(f"  [graph] {fname}")
        with open(os.path.join(src_dir, fname), encoding='utf-8') as f:
            json_trace = json.load(f)

        nodes = json_trace['nodes']
        id_to_node = {n['id']: n for n in nodes}

        nodes_level2 = [n for n in nodes if n.get('parent') == 2]
        nodes_main = [n for n in nodes if n.get('parent') == 1]

        back_id = None
        for n in nodes_main:
            if n['id'] not in {1, 2}:
                back_id = n['id']
        nodes_back = [n for n in nodes if n.get('parent') == back_id]

        def node_row(name, node, idx):
            return {
                'modelid': fname.replace('graph_', '').replace('.json', ''),
                'layerid': idx,
                'cpueventname': name,
                'inputshapes': json.dumps(node.get('input_shapes', [])),
                'inputvalues': json.dumps(node.get('inputs', [])),
                'inputtypes': json.dumps(node.get('input_types', [])),
                'outputshapes': json.dumps(node.get('output_shapes', [])),
                'outputvalues': json.dumps(node.get('outputs', [])),
                'outputtypes': json.dumps(node.get('output_types', [])),
                'op_schema': node.get('op_schema', '').replace(',', ';'),
            }

        def is_empty(n):
            return (n.get('input_shapes') == [] and n.get('inputs') == [] and
                    n.get('input_types') == [] and n.get('output_shapes') == [] and
                    n.get('outputs') == [] and n.get('output_types') == [])

        def get_children(node):
            return [n for n in nodes if n.get('parent') == node['id']]

        rows = []
        idx = 0
        optimizer_ids = []

        for n in nodes_level2:
            if 'Optimizer' in n['name']:
                optimizer_ids.append(n['id'])
            if not is_empty(n):
                idx += 1
                rows.append(node_row(n['name'], n, idx))

        for n in nodes_back:
            children = get_children(n)
            grandchildren = get_children(children[0]) if children else []
            target = grandchildren[0] if grandchildren else (children[0] if children else None)
            if target is None or is_empty(target):
                continue
            idx += 1
            rows.append(node_row(n['name'], target, idx))

        for op_id in optimizer_ids:
            for subnode in [n for n in nodes if n.get('parent') == op_id]:
                idx += 1
                rows.append(node_row(subnode['name'], subnode, idx))

        df = pd.DataFrame(rows)
        _validate_df(df, _GRAPH_COLS, f'_process_graph/{fname}')
        out_path = os.path.join(out_dir, fname.replace('.json', '.csv'))
        df.to_csv(out_path, index=False)


def _merge_data(config):
    """Merge graph and profiler CSVs; classify stage and tpflag."""
    graph_dir = os.path.join(config['data_dir'], 'middledata', 'graph')
    profiler_dir = os.path.join(config['data_dir'], 'middledata', 'profiler')
    out_dir = os.path.join(config['data_dir'], 'middledata', 'mergedata')
    os.makedirs(out_dir, exist_ok=True)

    target_ops = config.get('target_ops', DEFAULT_TARGET_OPS)

    for fname in os.listdir(graph_dir):
        if fname == '.DS_Store':
            continue
        profiler_fname = fname.replace('graph_', 'profiler_')
        profiler_path = os.path.join(profiler_dir, profiler_fname)
        if not os.path.exists(profiler_path):
            print(f"  [merge] WARNING: no matching profiler file for {fname}, skipping")
            continue
        print(f"  [merge] {fname}")

        df_graph = pd.read_csv(os.path.join(graph_dir, fname))
        df_profiler = pd.read_csv(profiler_path)

        merged = pd.merge(df_graph, df_profiler, on=['layerid', 'cpueventname'], suffixes=('_graph', '_profiler'))
        merged = merged[merged['cpueventname'] != 'aten::item']

        conditions = [
            merged['cpueventname'].str.startswith('autograd'),
            merged['cpueventname'].str.startswith('aten::_for'),
        ]
        merged['stage'] = np.select(conditions, ['backward', 'optimizer'], default='forward')

        target_sids = merged.loc[merged['cpueventname'].isin(target_ops), 'sequenceid'].unique()
        merged['tpflag'] = 0
        merged.loc[merged['sequenceid'].isin(target_sids), 'tpflag'] = 1

        _validate_df(merged, _MERGE_COLS, f'_merge_data/{fname}')
        merged.to_csv(os.path.join(out_dir, fname.replace('graph_', '')), index=False)


def _format_trace(config):
    """Convert merged CSVs to final tensor.csv and trace.csv for the simulator."""
    src_dir = os.path.join(config['data_dir'], 'middledata', 'mergedata')

    def parse_op_schema(schema_str):
        parts = schema_str.split(';')
        result = []
        for i, part in enumerate(parts):
            if i == 0:
                part = part.split('(')[1] if '(' in part else part
            elif i == len(parts) - 1:
                part = part.split(')')[0] if ')' in part else part
            result.append(part.strip().split(' '))
        return result

    def parse_generic_list_type(type_str):
        inner = type_str.replace('GenericList[', '').replace(']', '')
        return [t.strip() for t in inner.split(',')]

    for fname in os.listdir(src_dir):
        if fname == '.DS_Store':
            continue
        print(f"  [format] {fname}")
        df = pd.read_csv(os.path.join(src_dir, fname))

        tensor_rows = []
        operator_rows = []

        for _, row in df.iterrows():
            try:
                input_shapes = json.loads(row['inputshapes'])
                input_values = json.loads(row['inputvalues'])
                input_types = json.loads(row['inputtypes'])
                output_shapes = json.loads(row['outputshapes'])
                output_values = json.loads(row['outputvalues'])
                output_types = json.loads(row['outputtypes'])
            except (json.JSONDecodeError, TypeError) as e:
                print(f"    WARNING: JSON parse error in row {row.get('layerid')}: {e}, skipping")
                continue

            if pd.isna(row['op_schema']):
                continue
            schema_str = (row['op_schema']
                          .replace('Tensor(a!)', 'Tensor')
                          .replace('Tensor(a)', 'Tensor')
                          .replace('Tensor(a -> *) self', 'Tensor input')
                          .replace('Tensor self', 'Tensor input'))
            if not schema_str.strip():
                continue
            op_schema = parse_op_schema(schema_str)

            input_tensor_ids = []
            output_tensor_ids = []
            input_sizes = []
            output_sizes = []

            for i in range(len(input_values)):
                if i >= len(input_types):
                    break
                t = input_types[i]
                if 'tensor' not in t.lower():
                    continue
                if 'GenericList' not in t:
                    tid = input_values[i][0]
                    tensor_rows.append((
                        tid,
                        input_shapes[i] if i < len(input_shapes) else [],
                        input_values[i][3],
                        input_values[i][4],
                        op_schema[i][1].replace(')', '').replace('(', '') if i < len(op_schema) and len(op_schema[i]) > 1 else 'input',
                        input_values[i][1],
                    ))
                    input_tensor_ids.append(tid)
                    input_sizes.append(input_values[i][3])
                else:
                    sub_types = parse_generic_list_type(t)
                    val_list = input_values[i]
                    shape_list = input_shapes[i] if i < len(input_shapes) else []
                    for j, _ in enumerate(sub_types):
                        if j >= len(val_list):
                            break
                        tid = val_list[j][0]
                        tensor_rows.append((
                            tid,
                            shape_list[j] if j < len(shape_list) else [],
                            val_list[j][3],
                            val_list[j][4],
                            'input',
                            val_list[j][1],
                        ))
                        input_tensor_ids.append(tid)
                        input_sizes.append(val_list[j][3])

            for i in range(len(output_values)):
                if i >= len(output_types):
                    break
                t = output_types[i]
                if 'tensor' not in t.lower():
                    continue
                if 'GenericList' not in t:
                    tid = output_values[i][0]
                    tensor_rows.append((
                        tid,
                        output_shapes[i] if i < len(output_shapes) else [],
                        output_values[i][3],
                        output_values[i][4],
                        'output',
                        output_values[i][1],
                    ))
                    output_tensor_ids.append(tid)
                    output_sizes.append(output_values[i][3])
                else:
                    sub_types = parse_generic_list_type(t)
                    val_list = output_values[i]
                    shape_list = output_shapes[i] if i < len(output_shapes) else []
                    for j, _ in enumerate(sub_types):
                        if j >= len(val_list):
                            break
                        tid = val_list[j][0]
                        tensor_rows.append((
                            tid,
                            shape_list[j] if j < len(shape_list) else [],
                            val_list[j][3],
                            val_list[j][4],
                            'output',
                            val_list[j][1],
                        ))
                        output_tensor_ids.append(tid)
                        output_sizes.append(val_list[j][3])

            operator_rows.append((
                row['layerid'],
                row['cpueventname'],
                str(input_tensor_ids).replace(',', ';'),
                str(output_tensor_ids).replace(',', ';'),
                row['cudatime'],
                row['cudatimenooverlap'],
                str(input_sizes).replace(',', ';'),
                str(output_sizes).replace(',', ';'),
                '0',
                row['stage'],
                row['tpflag'],
            ))

        tensor_df = pd.DataFrame(tensor_rows,
                                 columns=['TensorID', 'TensorShape', 'TensorNumElement',
                                          'TensorEachByte', 'TensorType', 'TensorStorgeid'])
        tensor_df.insert(0, 'Index', range(len(tensor_df)))
        tensor_df['gpuid'] = '0'
        tensor_df = tensor_df[['Index', 'TensorID', 'TensorShape', 'TensorNumElement',
                                'TensorEachByte', 'TensorType', 'TensorStorgeid', 'gpuid']]

        op_df = pd.DataFrame(operator_rows,
                             columns=['OperatorID', 'OperatorName', 'Operator_input', 'Operator_output',
                                      'Operator_cudatime', 'Operator_cudatimenooverlap',
                                      'InputSize', 'OutputSize', 'gpuid', 'stage', 'tpflag'])
        op_df = op_df[op_df['Operator_cudatime'] != 0]

        model_name = re.sub(r'-iter.*', '', fname.replace('.csv', ''))
        out_dir = os.path.join(config['data_dir'], 'middledata', 'trace', model_name)
        os.makedirs(out_dir, exist_ok=True)
        tensor_df.to_csv(os.path.join(out_dir, 'tensor.csv'), index=False)
        op_df.to_csv(os.path.join(out_dir, 'trace.csv'), index=False)
        print(f"    wrote {len(tensor_df)} tensors, {len(op_df)} operators → {out_dir}")


def process_traces(config):
    data_dir = config.get('data_dir', DEFAULT_CONFIG['data_dir'])
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_SCRIPT_DIR, data_dir)
    config = {**config, 'data_dir': data_dir}

    print("==> Stage 1/4: processing profiler JSON files")
    _process_layer(config)
    print("==> Stage 2/4: processing graph JSON files")
    _process_graph(config)
    print("==> Stage 3/4: merging profiler and graph data")
    _merge_data(config)
    print("==> Stage 4/4: formatting final trace CSVs")
    _format_trace(config)

    out_dir = os.path.join(config['data_dir'], 'middledata', 'trace')
    print(f"\nDone. Traces written to: {out_dir}")


if __name__ == '__main__':
    import yaml

    cfg = DEFAULT_CONFIG.copy()

    config_path = os.path.join(_SCRIPT_DIR, 'trace_config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)
        for key in ('data_dir',):
            if key in file_cfg and not os.path.isabs(file_cfg[key]):
                cfg[key] = os.path.join(_SCRIPT_DIR, file_cfg[key])

    process_traces(cfg)
