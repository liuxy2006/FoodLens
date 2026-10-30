"""真实设备前向、反向与优化器更新；逐个数据类型报告失败原因。"""
import argparse
import json
from pathlib import Path
import time


def synchronize(device):
    import torch
    if device == 'xpu':
        torch.xpu.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def optimizer_probe(device, dtype):
    import torch
    start = time.perf_counter()
    parameter = torch.nn.Parameter(torch.randn(32, 32, device=device, dtype=getattr(torch, dtype)))
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    inputs = torch.randn(32, 32, device=device, dtype=getattr(torch, dtype))
    loss = (inputs @ parameter).float().square().mean()
    loss.backward()
    finite_gradient = bool(torch.isfinite(parameter.grad).all())
    optimizer.step()
    synchronize(device)
    return {'loss': loss.item(), 'finite_gradient': finite_gradient,
            'updated': bool((before != parameter).any()),
            'finite_parameters': bool(torch.isfinite(parameter).all()),
            'seconds': time.perf_counter() - start}


def probe():
    import torch
    from .environment import inspect_environment
    result = inspect_environment()
    result['torch_build'] = {'version': torch.__version__, 'cuda': torch.version.cuda,
                             'xpu': getattr(torch.version, 'xpu', None)}
    result['devices'] = {}
    for device, available in [('cpu', True), ('cuda', torch.cuda.is_available()),
                              ('xpu', hasattr(torch, 'xpu') and torch.xpu.is_available())]:
        details = {'available': available, 'tests': {}}
        if available:
            if device != 'cpu':
                backend = getattr(torch, device)
                details['name'] = backend.get_device_name(0)
                details['properties'] = str(backend.get_device_properties(0))
                try:
                    details['memory_free_total_bytes'] = list(backend.mem_get_info())
                except (AttributeError, RuntimeError) as exc:
                    details['memory_query_error'] = str(exc)
            for dtype in ('float32', 'bfloat16', 'float16'):
                try:
                    details['tests'][dtype] = optimizer_probe(device, dtype)
                except Exception as exc:
                    details['tests'][dtype] = {'error': f'{type(exc).__name__}: {exc}'}
        else:
            details['reason'] = 'backend reports unavailable; see torch build and recorded driver information'
        result['devices'][device] = details
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/stage2/device_probe.json')
    args = parser.parse_args()
    result = probe()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
