"""仅 train/val 的 LoRA 训练入口，默认离线；不会读 test。"""
import argparse
import json
from pathlib import Path
import random
import time
from .data import DietDataset, image_path
from .vlm import SupervisedCollator, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='task2_diet_record/train_config.json')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--model-path', help='本地已下载模型目录，覆盖配置中的模型名')
    parser.add_argument('--revision', help='下载时必须指定完整模型 commit SHA')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'xpu'], default='cpu')
    parser.add_argument('--output', default='outputs/smolvlm_lora')
    parser.add_argument('--max-steps', type=int, default=None, help='快速测试的优化器步数上限')
    parser.add_argument('--val-limit', type=int, default=None, help='快速测试专用；正式实验省略')
    parser.add_argument('--generate-val', type=int, default=0, help='额外生成前 N 个验证样本，0 表示仅算验证损失')
    args = parser.parse_args()
    if args.allow_download and (not args.revision or len(args.revision) != 40):
        parser.error('online download requires a full 40-character revision SHA')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if any(config[key] <= 0 for key in ('epochs', 'batch_size', 'gradient_accumulation', 'learning_rate')):
        parser.error('training sizes and learning rate must be positive')
    if args.max_steps is not None and args.max_steps <= 0 or args.val_limit is not None and args.val_limit <= 0:
        parser.error('smoke-test limits must be positive')
    import numpy as np
    try:
        import torch
        import peft
        import transformers
    except ImportError as exc:
        raise SystemExit('Missing training dependency: ' + str(exc) +
                         '. See task2_diet_record/README.md; no packages were installed.') from exc
    from torch.utils.data import DataLoader, Subset
    from peft import LoraConfig, get_peft_model
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = Path(args.data_root)
    train = DietDataset(root / 'raw/annotations/train.jsonl', root, 'train')
    val = DietDataset(root / 'raw/annotations/val.jsonl', root, 'val')
    if not len(train) or not len(val):
        raise ValueError('empty train or validation set')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new output folder to preserve previous experiments')
    model_path = args.model_path or config['model']
    model, processor = load_backbone(model_path, args.device, args.revision, args.allow_download)
    # 只给语言模块的注意力投影加 LoRA，视觉编码器保持冻结。
    targets = [name for name, _ in model.named_modules()
               if 'text_model' in name and name.endswith(('.q_proj', '.v_proj'))]
    if not targets:
        raise ValueError('no supported SmolVLM text attention targets found; inspect architecture')
    model = get_peft_model(model, LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'],
                          lora_dropout=config['lora_dropout'], target_modules=targets,
                          task_type='CAUSAL_LM', bias='none'))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.print_trainable_parameters()
    collate = SupervisedCollator(processor, config['max_length'])
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, collate_fn=collate,
                        generator=generator, num_workers=0)
    val_subset = Subset(val, range(min(args.val_limit, len(val)))) if args.val_limit else val
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
    output.mkdir(parents=True)
    import importlib.metadata
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ('torch', 'transformers', 'peft', 'numpy')}
    import PIL
    versions['Pillow'] = PIL.__version__
    run = {'config': config, 'arguments': vars(args), 'versions': versions,
           'resolved_model_revision': getattr(model.config, '_commit_hash', None),
           'train_records': len(train), 'val_records': len(val_subset)}
    import hashlib
    run['annotation_sha256'] = {split: hashlib.sha256((root / f'raw/annotations/{split}.jsonl').read_bytes()).hexdigest()
                                for split in ('train', 'val')}
    (output / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    step, best = 0, float('inf')
    for epoch in range(config['epochs']):
        model.train(); optimizer.zero_grad(); train_loss = 0.0; batches = 0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            # 最后一组不足 accumulation 时，按真实组大小缩放。
            accumulation = config['gradient_accumulation']
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            train_loss += loss.item(); batches += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(); step += 1
                if args.max_steps and step >= args.max_steps:
                    break
        model.eval(); val_loss = 0.0; token_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(args.device) for key, value in batch.items()}
                count = (batch['labels'][:, 1:] != -100).sum().item()
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite validation loss')
                val_loss += loss.item() * count; token_count += count
        log = {'epoch': epoch + 1, 'optimizer_steps': step, 'train_loss': train_loss / batches,
               'val_loss': val_loss / token_count, 'time': time.time()}
        if args.generate_val:
            from .interfaces import TransformersDietModel
            from .predict import predict_records
            from evaluation.diet_metrics import evaluate_predictions
            rows = val.records[:args.generate_val]
            predictor = TransformersDietModel(model, processor, config['max_new_tokens'])
            predictions = predict_records(predictor, rows, root)
            log['generation_metrics'] = evaluate_predictions([p['prediction'] for p in predictions], [r['target'] for r in rows])
            (output / f'val_epoch_{epoch+1}.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions), encoding='utf-8')
        if log['val_loss'] < best:
            best = log['val_loss']
            model.save_pretrained(output / 'best'); processor.save_pretrained(output / 'best')
        model.save_pretrained(output / 'last'); processor.save_pretrained(output / 'last')
        torch.save({'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'step': step,
                    'torch_rng': torch.get_rng_state()}, output / 'optimizer.pt')
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(log) + '\n')
        print(json.dumps(log))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
