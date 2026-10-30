"""检查环境，不联网、不安装依赖；缺失项明确记为 unavailable。"""
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil


def inspect_environment():
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {}}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'Pillow', 'numpy', 'scipy', 'jsonschema', 'psutil'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    # 某些 Conda 环境缺少 wheel 元信息，但 Pillow 本身仍然可以正常导入。
    if result['packages']['Pillow'] is None:
        try:
            import PIL
            result['packages']['Pillow'] = PIL.__version__
        except ImportError:
            pass
    try:
        import psutil
        memory = psutil.virtual_memory()
        result['ram_bytes'] = {'total': memory.total, 'available': memory.available}
    except ImportError:
        result['ram_bytes'] = None
    usage = shutil.disk_usage('.')
    result['disk_bytes'] = {'total': usage.total, 'free': usage.free}
    result['display_adapters'] = []
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Display(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('name', wintypes.WCHAR * 32),
                        ('description', wintypes.WCHAR * 128), ('flags', wintypes.DWORD),
                        ('id', wintypes.WCHAR * 128), ('key', wintypes.WCHAR * 128)]
        display = Display(); display.cb = ctypes.sizeof(display); i = 0
        while ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(display), 0):
            if display.description not in result['display_adapters']:
                result['display_adapters'].append(display.description)
            i += 1
    try:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        result['cuda_gpus'] = [{'name': torch.cuda.get_device_name(i),
                               'vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                              for i in range(torch.cuda.device_count())]
        result['xpu_available'] = hasattr(torch, 'xpu') and torch.xpu.is_available()
    except ImportError:
        result.update(cuda_available=None, xpu_available=None, cuda_gpus=None)
    return result


if __name__ == '__main__':
    result = inspect_environment()
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/environment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
