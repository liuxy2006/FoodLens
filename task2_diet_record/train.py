"""仅 train/val 的 LoRA 训练入口，默认离线；不会读 test。"""
import argparse
import json
from pathlib import Path
import random
import time
from .data import DietDataset
from .vlm import SupervisedCollator, load_backbone, configure_lora
from .checkpoint import (file_sha256, inspect_checkpoint, validate_resume_signature,
                         save_checkpoint, restore_training_state)


def verify_update(parameters, before):
    """真实梯度/参数检查，不能只凭 loss 变化声称更新成功。"""
    import torch
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters.values())
    changed = [name for name, parameter in parameters.items()
               if not torch.equal(parameter.detach().cpu(), before[name])]
    if not finite or not changed:
        raise FloatingPointError('nonfinite parameters or no trainable parameter updated')
    return {'finite_parameters': finite, 'updated_parameter_tensors': len(changed)}


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
    parser.add_argument('--train-limit', type=int, help='仅用于集成测试的训练样本数')
    parser.add_argument('--resume', help='完整 checkpoint-xxxxxx 目录；恢复到新的 output 目录')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--verify-update', action='store_true')
    parser.add_argument('--verify-reload', action='store_true', help='保存前后对同一验证图片作确定性推理')
    args = parser.parse_args()
    if args.allow_download and (not args.revision or len(args.revision) != 40):
        parser.error('online download requires a full 40-character revision SHA')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if any(config[key] <= 0 for key in ('epochs', 'batch_size', 'gradient_accumulation', 'learning_rate')):
        parser.error('training sizes and learning rate must be positive')
    if args.max_steps is not None and args.max_steps <= 0 or args.val_limit is not None and args.val_limit <= 0:
        parser.error('smoke-test limits must be positive')
    if args.train_limit is not None and args.train_limit <= 0 or args.checkpoint_every <= 0 or args.generate_val < 0:
        parser.error('invalid sample/checkpoint limits')
    import numpy as np
    try:
        import torch
        import peft
        import transformers
    except ImportError as exc:
        raise SystemExit('Missing training dependency: ' + str(exc) +
                         '. See task2_diet_record/README.md; no packages were installed.') from exc
    from torch.utils.data import DataLoader, Subset
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = Path(args.data_root)
    train = DietDataset(root / 'raw/annotations/train.jsonl', root, 'train')
    val = DietDataset(root / 'raw/annotations/val.jsonl', root, 'val')
    if args.train_limit:
        train.records = train.records[:args.train_limit]
    if not len(train) or not len(val):
        raise ValueError('empty train or validation set')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new output folder to preserve previous experiments')
    model_path = args.model_path or config['model']
    hashes = {split: file_sha256(root / f'raw/annotations/{split}.jsonl') for split in ('train', 'val')}
    model_config = Path(model_path) / 'config.json'
    signature = {'config': config, 'annotation_sha256': hashes, 'train_records': len(train),
                 'val_limit': args.val_limit, 'model_revision': args.revision,
                 'model_config_sha256': file_sha256(model_config) if model_config.is_file() else None}
    saved = inspect_checkpoint(args.resume) if args.resume else None
    if saved:
        validate_resume_signature(saved['signature'], signature)
        if args.max_steps and args.max_steps <= saved['optimizer_steps']:
            parser.error('--max-steps is a TOTAL step limit and must exceed restored optimizer_steps')
    model, processor = load_backbone(model_path, args.device, args.revision, args.allow_download,
                                     processor_path=args.resume)
    model, targets = configure_lora(model, config, args.resume)
    model.print_trainable_parameters()
    collate = SupervisedCollator(processor, config['max_length'])
    val_subset = Subset(val, range(min(args.val_limit, len(val)))) if args.val_limit else val
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], collate_fn=collate, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
    if saved:
        restore_training_state(args.resume, optimizer)
    output.mkdir(parents=True)
    import importlib.metadata
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ('torch', 'transformers', 'peft', 'numpy')}
    import PIL
    versions['Pillow'] = PIL.__version__
    run = {'config': config, 'arguments': vars(args), 'versions': versions,
           'resolved_model_revision': getattr(model.config, '_commit_hash', None),
           'train_records': len(train), 'val_records': len(val_subset), 'signature': signature,
           'model_parameters': sum(p.numel() for p in model.parameters()),
           'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
           'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
           'lora_targets': targets}
    if args.verify_update:
        run['all_model_parameters_finite_before_training'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        if not run['all_model_parameters_finite_before_training']:
            raise FloatingPointError('model has nonfinite parameters before training')
    import subprocess
    run['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    step = saved['optimizer_steps'] if saved else 0
    best = saved['best_val_loss'] if saved else None
    best_adapter = saved.get('best_adapter') if saved else None
    start_epoch = saved['next_epoch'] if saved else 0
    start_sample = saved['next_sample'] if saved else 0
    if start_epoch >= config['epochs']:
        raise ValueError('checkpoint already completed configured epochs')
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    for epoch in range(start_epoch, config['epochs']):
        # 每轮顺序由独立 generator 决定，恢复时跳过已消费样本，不扰动模型 RNG。
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
        offset = start_sample if epoch == start_epoch else 0
        loader = DataLoader(Subset(train, order[offset:]), batch_size=config['batch_size'], collate_fn=collate,
                            generator=torch.Generator().manual_seed(config['seed'] + epoch), num_workers=0)
        model.train(); optimizer.zero_grad(); train_loss = 0.0; batches = 0
        group_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index % config['gradient_accumulation'] == 0:
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                group_started = time.perf_counter()
                before = {name: p.detach().cpu().clone() for name, p in parameters.items()} if args.verify_update else None
            batch = {key: value.to(args.device) for key, value in batch.items()}
            if args.verify_update and batch_index == 0:
                image_token_id = getattr(model.config, 'image_token_id', None)
                image_mask = batch['input_ids'] == image_token_id if image_token_id is not None else None
                if 'pixel_values' not in batch or image_mask is None or not bool(image_mask.any()):
                    raise ValueError('missing image tensor or image tokens in multimodal batch')
                if bool((batch['labels'][image_mask] != -100).any()):
                    raise ValueError('image special tokens must not be supervised')
                inspection = {'pixel_values_shape': list(batch['pixel_values'].shape),
                              'image_token_count': int(image_mask.sum().item()),
                              'supervised_token_count': int((batch['labels'] != -100).sum().item()),
                              'masked_token_count': int((batch['labels'] == -100).sum().item())}
                (output / f'batch_inspection_epoch_{epoch+1}.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
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
                gradient_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters.values())
                nonzero_gradient = any(p.grad is not None and bool((p.grad != 0).any()) for p in parameters.values())
                if not gradient_finite or not nonzero_gradient:
                    raise FloatingPointError('nonfinite or all-zero LoRA gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(); step += 1
                diagnostics = verify_update(parameters, before) if args.verify_update else {}
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                diagnostics.update(step=step, loss=loss.item(), finite_gradient=gradient_finite,
                                   nonzero_gradient=nonzero_gradient, seconds=time.perf_counter() - group_started)
                if args.device == 'cuda':
                    diagnostics['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    diagnostics['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(diagnostics) + '\n')
                next_sample = min(offset + (batch_index + 1) * config['batch_size'], len(train))
                next_epoch = epoch + 1 if next_sample == len(train) else epoch
                cursor = 0 if next_sample == len(train) else next_sample
                terminal = batch_index + 1 == len(loader) or (args.max_steps and step >= args.max_steps)
                if step % args.checkpoint_every == 0 and not terminal:
                    checkpoint = output / f'checkpoint-{step:06d}'
                    save_checkpoint(checkpoint, model, processor, optimizer, {
                        'signature': signature, 'optimizer_steps': step, 'next_epoch': next_epoch,
                        'next_sample': cursor, 'best_val_loss': best, 'best_adapter': best_adapter})
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
        if best is None or log['val_loss'] < best:
            best = log['val_loss']
            model.save_pretrained(output / 'best'); processor.save_pretrained(output / 'best')
            best_adapter = str((output / 'best').resolve())
        checkpoint = output / f'checkpoint-{step:06d}'
        save_checkpoint(checkpoint, model, processor, optimizer, {
            'signature': signature, 'optimizer_steps': step, 'next_epoch': next_epoch,
            'next_sample': cursor, 'best_val_loss': best, 'best_adapter': best_adapter})
        (output / 'last_checkpoint.json').write_text(json.dumps({'path': str(checkpoint.resolve())}), encoding='utf-8')
        (output / 'best_adapter.json').write_text(json.dumps({'path': best_adapter}), encoding='utf-8')
        if args.verify_reload:
            from .integration import verify_adapter_reload
            log['reload'] = verify_adapter_reload(model, processor, model_path, checkpoint,
                                                  val.records[0], root, args.device, config['max_new_tokens'])
            (output / f'reload_check_{step:06d}.json').write_text(json.dumps(log['reload'], indent=2), encoding='utf-8')
            if not log['reload']['identical_text']:
                raise RuntimeError('deterministic generation differs after adapter reload')
        if args.device == 'cuda':
            log['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            log['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        import psutil
        log['process_rss_bytes'] = psutil.Process().memory_info().rss
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(log) + '\n')
        print(json.dumps(log))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
