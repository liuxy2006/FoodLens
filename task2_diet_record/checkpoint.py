"""不可覆盖的训练断点，完整性校验通过后才允许恢复。"""
import hashlib
import json
from pathlib import Path
import random


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_signature(saved, current):
    if saved != current:
        different = [key for key in set(saved) | set(current) if saved.get(key) != current.get(key)]
        raise ValueError(f'resume configuration/data/model mismatch: {sorted(different)}')


def inspect_checkpoint(path):
    """纯文件检查，无须加载 torch；断电/Drive 写入不完整的目录不可恢复。"""
    path = Path(path)
    manifest = json.loads((path / 'READY.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(path.resolve()) or file_sha256(candidate) != expected:
            raise ValueError(f'checkpoint integrity mismatch: {relative}')
    required = {'training_state.pt', 'metadata.json', 'adapter_config.json', 'adapter_model.safetensors',
                'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('incomplete checkpoint manifest')
    return json.loads((path / 'metadata.json').read_text(encoding='utf-8'))


def save_checkpoint(path, model, processor, optimizer, metadata):
    import numpy as np
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    np_state = np.random.get_state()
    state = {'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
             'python_rng': random.getstate(),
             'numpy_rng': [np_state[0], np_state[1].tolist(), *np_state[2:]],
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state, path / 'training_state.pt')
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    files = {p.relative_to(path).as_posix(): file_sha256(p) for p in path.rglob('*') if p.is_file()}
    # READY 最后写入，避免把尚未保存完整的断点当成成功。
    (path / 'READY.json').write_text(json.dumps({'files': files}, indent=2), encoding='utf-8')


def restore_training_state(path, optimizer):
    import numpy as np
    import torch
    # 只恢复自己生成并经完整性核对的断点，使用受限的 weights_only 加载。
    state = torch.load(Path(path) / 'training_state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np_state = state['numpy_rng']
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
