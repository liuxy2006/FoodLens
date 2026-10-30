"""只安装 Colab Python 依赖，禁止 pip 替换已经正常工作的 CUDA PyTorch。"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def torch_snapshot():
    # 新进程避免当前 notebook 缓存的 torch 掩盖环境被替换的问题。
    code = ('import torch,json; print(json.dumps({"version":torch.__version__, '
            '"cuda":torch.version.cuda,"available":torch.cuda.is_available()}))')
    return json.loads(subprocess.check_output([sys.executable, '-c', code], text=True))


def check_cuda_unchanged(before, after):
    if not before['available'] or not before['cuda']:
        raise RuntimeError('Colab CUDA PyTorch is not working before installation; select a GPU runtime')
    if not after['available'] or not after['cuda'] or before != after:
        raise RuntimeError(f'CUDA PyTorch changed or became unavailable after installation; stop and reset runtime. Before={before}, after={after}')


def check_install_plan(report):
    for entry in report.get('install', []):
        name = entry['metadata']['name'].lower().replace('_', '-')
        url = entry.get('download_info', {}).get('url', '').lower()
        if (name in {'torch', 'torchvision', 'torchaudio', 'triton'}
                or name.startswith(('nvidia-', 'intel-', 'onemkl', 'dpcpp-', 'pytorch-triton'))
                or 'win_amd64' in url or '+xpu' in url or '%2bxpu' in url):
            raise RuntimeError(f'pip plan would install/replace GPU runtime package: {name}; no installation performed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    before = torch_snapshot()
    print('Before installation:', json.dumps(before), flush=True)
    check_cuda_unchanged(before, before)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'torch_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    constraints = []
    for name in ('torch', 'torchvision', 'torchaudio', 'triton'):
        try:
            constraints.append(f'{name}==={importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            pass
    constraints_path = output / 'preserve_torch.txt'
    constraints_path.write_text('\n'.join(constraints) + '\n', encoding='utf-8')
    command = [sys.executable, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
               '--constraint', str(constraints_path), '-r', 'requirements.txt',
               '-r', 'task2_diet_record/requirements-train.txt']
    plan_path = output / 'pip_plan.json'
    subprocess.run([*command, '--dry-run', '--report', str(plan_path)], check=True)
    check_install_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    try:
        subprocess.run([*command, '--report', str(output / 'pip_install.json')], check=True)
    finally:
        try:
            after = torch_snapshot()
        except Exception as exc:
            (output / 'torch_after_error.txt').write_text(str(exc), encoding='utf-8')
            raise RuntimeError('Cannot import/check CUDA PyTorch after dependency installation; stop and reset Colab runtime') from exc
        (output / 'torch_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
        print('After installation:', json.dumps(after), flush=True)
        check_cuda_unchanged(before, after)
    subprocess.run([sys.executable, '-c',
                    'import transformers,peft,accelerate; '
                    'from transformers import AutoProcessor,AutoModelForVision2Seq,Idefics3ForConditionalGeneration; '
                    'print(transformers.__version__,peft.__version__,accelerate.__version__)'], check=True)
    (output / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')


if __name__ == '__main__':
    main()
