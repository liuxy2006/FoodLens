"""Colab 独立入口：16 图，先 25 步，人工审查后最多到 100 步。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .checkpoint import inspect_checkpoint
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, choices=[25, 100], default=25)
    parser.add_argument('--resume')
    args = parser.parse_args()
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
