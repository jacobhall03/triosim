import os
import sys
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.profiler import ExecutionTraceObserver

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
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
}


def collect_traces(config):
    device = torch.device("cuda:0")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    graph_dir = os.path.join(config['data_dir'], 'graph')
    profiler_dir = os.path.join(config['data_dir'], 'profiler')
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(profiler_dir, exist_ok=True)

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_iters = config.get('num_iters', 43)
    profile_start = config.get('profile_start', 40)
    warmup_start = config.get('warmup_start', 30)

    for bs in config['batch_sizes']:
        dataset = torchvision.datasets.ImageFolder(config['dataset_path'], data_transforms)
        subset = torch.utils.data.Subset(dataset, range(bs))
        loader = torch.utils.data.DataLoader(
            subset, batch_size=bs, shuffle=False,
            num_workers=config.get('workers', 8),
        )

        for model_name in config['models']:
            print(f"\nCollecting trace: model={model_name}, batch_size={bs}")
            try:
                model_fn = getattr(models, model_name)
                model = model_fn(weights=None).to(device)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                model.train()

                for batch in tqdm(loader, total=len(loader)):
                    features, labels = batch[0].to(device), batch[1].to(device)
                    print(f"  batch shape: {features.size()}")
                    total_time_profiled = 0.0
                    total_time_warmup = 0.0

                    for i in range(num_iters):
                        if i > profile_start:
                            graph_path = os.path.join(graph_dir, f"graph_{model_name}-iter{i}.json")
                            profiler_path = os.path.join(profiler_dir, f"profiler_{model_name}-iter{i}.json")
                            eg = ExecutionTraceObserver()
                            eg.register_callback(graph_path)
                            eg.start()
                            with profile(
                                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                record_shapes=True, with_stack=True, profile_memory=True,
                            ) as prof:
                                torch.cuda.synchronize()
                                starter = torch.cuda.Event(enable_timing=True)
                                ender = torch.cuda.Event(enable_timing=True)
                                starter.record()
                                optimizer.zero_grad()
                                preds = model(features)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                                ender.record()
                                torch.cuda.synchronize()
                                total_time_profiled += starter.elapsed_time(ender)
                            prof.export_chrome_trace(profiler_path)
                            eg.stop()
                            eg.unregister_callback()
                        elif i > warmup_start:
                            torch.cuda.synchronize()
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            starter.record()
                            optimizer.zero_grad()
                            preds = model(features)
                            loss = loss_fn(preds, labels)
                            loss.backward()
                            optimizer.step()
                            ender.record()
                            torch.cuda.synchronize()
                            total_time_warmup += starter.elapsed_time(ender)
                        else:
                            optimizer.zero_grad()
                            preds = model(features)
                            loss = loss_fn(preds, labels)
                            loss.backward()
                            optimizer.step()

                    profiled_count = num_iters - 1 - profile_start
                    warmup_count = profile_start - warmup_start
                    if profiled_count > 0:
                        print(f"  avg profiled iter: {total_time_profiled / profiled_count:.2f} ms")
                    if warmup_count > 0:
                        print(f"  avg warmup iter:   {total_time_warmup / warmup_count:.2f} ms")

            except Exception as e:
                print(f"  ERROR tracing {model_name}: {e}")


if __name__ == '__main__':
    import yaml

    cfg = DEFAULT_CONFIG.copy()

    config_path = os.path.join(_SCRIPT_DIR, 'trace_config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)
        for key in ('dataset_path', 'data_dir'):
            if key in file_cfg and not os.path.isabs(file_cfg[key]):
                cfg[key] = os.path.join(_SCRIPT_DIR, file_cfg[key])

    if len(sys.argv) > 1:
        cfg['batch_sizes'] = [int(sys.argv[1])]

    collect_traces(cfg)
