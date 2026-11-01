"""Colab 独立入口：16 图，先 25 步，人工审查后最多到 100 步。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .checkpoint import inspect_checkpoint, file_sha256, validate_resume_signature
from .interfaces import PROMPT
from .monitoring import overfit_decision


def check_stage(stage, resume_metadata=None):
    if stage not in (25, 100):
        raise ValueError('only the 25-step review stage and 100-step cap are allowed')
    if resume_metadata:
        signature = resume_metadata['signature']
        if signature['config'].get('experiment_kind') != 'schema_overfit' or signature['train_records'] != 16:
            raise ValueError('cannot restore smoke/full-data checkpoints into the schema experiment')
        completed = resume_metadata['optimizer_steps']
        if completed >= stage or stage == 100 and completed < 25:
            raise ValueError('review at least 25 steps before continuing; total limit must exceed saved steps')
    elif stage == 100:
        raise ValueError('run and review a fresh 25-step experiment before resuming to 100')


def completed_stage25(experiment_dir, model_path, revision, data_root):
    """只读核验已完成的 25 步目录；不训练、不加载权重、不修改原实验。"""
    directory = Path(experiment_dir).resolve()
    for relative in ('last_checkpoint.json', 'run.json', 'steps.jsonl', 'monitor/step-000025.json'):
        if not (directory / relative).is_file():
            raise FileNotFoundError(f'completed 25-step experiment is missing {relative}: {directory}')
    pointer = json.loads((directory / 'last_checkpoint.json').read_text(encoding='utf-8'))
    checkpoint = Path(pointer['path'])
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(directory):
        raise ValueError('25-step checkpoint pointer must stay inside the selected experiment directory')
    metadata = inspect_checkpoint(checkpoint)
    if metadata.get('optimizer_steps') != 25:
        raise ValueError('selected experiment must have completed exactly 25 steps; use --resume for a partial run')
    check_stage(100, metadata)
    root = Path(data_root)
    expected = {
        'config': json.loads(Path(__file__).with_name('overfit_config.json').read_text(encoding='utf-8')),
        'annotation_sha256': {split: file_sha256(root / f'raw/annotations/{split}.jsonl') for split in ('train', 'val')},
        'train_records': 16, 'val_limit': 4, 'model_revision': revision,
        'model_config_sha256': file_sha256(Path(model_path) / 'config.json'),
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'monitor_protocol': {'generate_train': 16, 'generate_val': 4, 'eval_every': 25}}
    validate_resume_signature(metadata['signature'], expected)
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    validate_resume_signature(run['signature'], expected)
    report = json.loads((directory / 'monitor/step-000025.json').read_text(encoding='utf-8'))
    if report.get('step') != 25:
        raise ValueError('the selected review report does not describe step 25')
    return {'experiment_dir': str(directory), 'checkpoint': str(checkpoint), 'optimizer_steps': 25}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, choices=[25, 100], default=25)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume')
    resume.add_argument('--existing-stage25-dir', help='已完成的 schema_step25 目录，只读复用后继续至总计 100 步')
    args = parser.parse_args()
    if args.existing_stage25_dir:
        if args.stage != 100:
            parser.error('--existing-stage25-dir is only for continuation to total step 100')
        selected = completed_stage25(args.existing_stage25_dir, args.model_path, args.revision, args.data_root)
        args.resume = selected['checkpoint']
    metadata = inspect_checkpoint(args.resume) if args.resume else None
    check_stage(args.stage, metadata)
    if Path(args.output).exists():
        raise FileExistsError('choose a NEW experiment directory; previous results are protected')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('run this entry in Colab NVIDIA GPU; no local CPU training fallback')
    command = [sys.executable, '-m', 'task2_diet_record.train', '--config',
               str(Path(__file__).with_name('overfit_config.json')), '--model-path', args.model_path,
               '--revision', args.revision, '--data-root', args.data_root, '--device', 'cuda',
               '--train-limit', '16', '--val-limit', '4', '--generate-train', '16', '--generate-val', '4',
               '--eval-every', '25', '--checkpoint-every', '25', '--max-steps', str(args.stage),
               '--verify-update', '--output', args.output]
    command += ['--resume', args.resume] if args.resume else ['--evaluate-start']
    subprocess.run(command, check=True)
    report = json.loads((Path(args.output) / 'monitor' / f'step-{args.stage:06d}.json').read_text(encoding='utf-8'))
    rate = report['train']['metrics']['schema_valid_rate']
    decision = {'step': args.stage, 'train_schema_rate': rate,
                'decision': overfit_decision(args.stage, rate), 'generalization_claim': False}
    (Path(args.output) / 'decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
