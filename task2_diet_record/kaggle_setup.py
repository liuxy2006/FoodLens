"""Kaggle 入口的挂载路径、旧断点迁移和 CUDA 预检。"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil


ARCHIVES = ('annotations.zip', 'images.zip')
STEP25_CHECKPOINT = 'checkpoint-000025'


def mounted_directory(input_root, selected):
    """只读取挂载目录，拒绝绝对路径、相对路径或软链接绕出挂载点。"""
    root = Path(input_root).resolve()
    directory = (root / selected).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f'输入目录必须在 {root} 内：{directory}')
    if not directory.is_dir():
        raise FileNotFoundError(f'找不到 Kaggle 挂载目录：{directory}；请先在 Add Input 中挂载私有 Dataset')
    return directory


def mounted_dataset_dir(input_root, selected=None):
    """两个 ZIP 应在同一目录；发现多个数据版本时要求手工选择。"""
    root = mounted_directory(input_root, '.')
    if selected:
        directory = mounted_directory(root, selected)
        missing = [name for name in ARCHIVES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f'挂载目录 {directory} 缺少：{", ".join(missing)}')
    else:
        candidates = sorted({path.parent for path in root.rglob(ARCHIVES[0])
                             if path.is_file() and (path.parent / ARCHIVES[1]).is_file()})
        if not candidates:
            raise FileNotFoundError(f'在 {root} 下没有找到同目录的 annotations.zip 和 images.zip；'
                                    '请在 Add Input 中挂载包含这两个 ZIP 的私有 Dataset')
        if len(candidates) != 1:
            raise ValueError(f'找到多个 ZIP 目录，请设置 DATASET_DIR：{", ".join(map(str, candidates))}')
        directory = mounted_directory(root, candidates[0])
    if any(not (directory / name).resolve().is_relative_to(root) for name in ARCHIVES):
        raise ValueError(f'ZIP 文件不得指向 {root} 之外')
    return directory


def stage25_for_kaggle(input_root, selected, destination):
    """只改临时副本的指针；现有诊断模块继续校验断点、签名和数据。"""
    source = mounted_directory(input_root, selected)
    pointer_file = source / 'last_checkpoint.json'
    if not pointer_file.is_file():
        raise FileNotFoundError(f'旧实验缺少 {pointer_file}；请上传完整 schema_step25 目录')
    pointer = PurePosixPath(json.loads(pointer_file.read_text(encoding='utf-8'))['path'])
    # Colab 保存的是绝对路径，迁移时只允许定位训练代码保存的最终 25 步目录。
    if (pointer.name != STEP25_CHECKPOINT or '..' in pointer.parts
            or (not pointer.is_absolute() and pointer != PurePosixPath(STEP25_CHECKPOINT))):
        raise ValueError(f'旧实验指针必须指向 {STEP25_CHECKPOINT}：{pointer}')
    checkpoint = mounted_directory(source, STEP25_CHECKPOINT)
    destination = Path(destination).resolve()
    if destination.is_relative_to(Path(input_root).resolve()):
        raise ValueError('旧实验临时副本必须位于 Kaggle 挂载目录之外')
    reports = ('run.json', 'steps.jsonl', 'monitor/step-000025.json')
    for relative in reports:
        report = source / relative
        if not report.is_file() or not report.resolve().is_relative_to(source):
            raise FileNotFoundError(f'旧实验缺少有效报告：{report}；请上传完整 schema_step25 目录')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checkpoint, destination / STEP25_CHECKPOINT)
    for relative in reports:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    (destination / pointer_file.name).write_text(json.dumps({'path': STEP25_CHECKPOINT}), encoding='utf-8')
    return destination


def probe_cuda():
    """实际执行与现有推理同精度的 CUDA 运算；不可用时不回退到 CPU。"""
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            raise RuntimeError(f'torch={torch.__version__}, build CUDA={torch.version.cuda}')
        tensor = torch.ones((2, 2), dtype=torch.float32, device='cuda:0')
        if (tensor @ tensor)[0, 0].item() != 2.0:
            raise RuntimeError('CUDA float32 matrix multiplication returned an unexpected result')
        properties = torch.cuda.get_device_properties(0)
        return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': True,
                'device': 'cuda:0', 'device_count': torch.cuda.device_count(),
                'gpu': properties.name, 'memory_bytes': properties.total_memory,
                'compute_capability': list(torch.cuda.get_device_capability(0)),
                'float32_matmul_ok': True}
    except Exception as exc:
        raise RuntimeError('Kaggle CUDA 预检失败：请选择 GPU T4 或重置 GPU 会话；不会回退到 CPU。'
                           f' 原因：{exc}') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-report', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = probe_cuda()
    except RuntimeError as exc:
        args.cuda_report.write_text(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    args.cuda_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
