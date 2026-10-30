"""固定图片的生成监控；训练集分数只代表记忆能力，不代表泛化。"""
from collections import Counter
import json
import math
import time
from pathlib import Path
from .schema import NUMERIC_FIELDS, parse_output, validate_record
from .interfaces import TransformersDietModel
from .predict import predict_records
from evaluation.diet_metrics import evaluate_predictions


def schema_observations(text):
    result = {'json_parsed': False, 'schema_valid': False}
    try:
        record = parse_output(text)
        result['json_parsed'] = True
        result['schema_errors'] = validate_record(record)
        result['schema_valid'] = not result['schema_errors']
        if isinstance(record, dict):
            result['top_level_keys'] = sorted(record)
            result['unexpected_top_level_keys'] = sorted(set(record) - {'items', *NUMERIC_FIELDS})
            result['missing_top_level_keys'] = sorted({'items', *NUMERIC_FIELDS} - set(record))
            result['items_is_array'] = isinstance(record.get('items'), list)
            result['numeric_top_level_fields'] = [field for field in NUMERIC_FIELDS
                if type(record.get(field)) in (int, float) and math.isfinite(record[field])]
            result['unexpected_item_keys'] = sorted({key for item in record.get('items', [])
                if isinstance(item, dict) for key in item if key not in ('name', 'weight_g')}) if result['items_is_array'] else []
            if result['items_is_array']:
                result['item_count'] = len(record['items'])
                result['numeric_item_weight_count'] = sum(
                    isinstance(item, dict) and type(item.get('weight_g')) in (int, float)
                    and math.isfinite(item['weight_g']) for item in record['items'])
    except (ValueError, TypeError) as exc:
        result['parse_error'] = str(exc)
    return result


def evaluate_split(predictor, rows, root, split):
    if split not in ('train', 'val') or not rows or any(row['split'] != split for row in rows):
        raise ValueError('monitoring only accepts nonempty matching train/val rows')
    predictions = predict_records(predictor, rows, root, include_generation_info=True)
    unexpected = Counter()
    for prediction, row in zip(predictions, rows):
        prediction['schema'] = schema_observations(prediction['prediction'])
        prediction['target'] = row['target']  # 仅在生成结束后附到报告，不传给模型。
        unexpected.update(prediction['schema'].get('unexpected_top_level_keys', []))
        unexpected.update(prediction['schema'].get('unexpected_item_keys', []))
    metrics = evaluate_predictions([p['prediction'] for p in predictions], [row['target'] for row in rows])
    return {'scope': 'train_memorization_only' if split == 'train' else 'fixed_val_subset_not_full_validation',
            'dish_ids': [row['dish_id'] for row in rows], 'metrics': metrics,
            'unexpected_key_occurrences': dict(unexpected), 'predictions': predictions}


def validation_loss(model, loader, device):
    import torch
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = (batch['labels'][:, 1:] != -100).sum().item()
            loss = model(**batch).loss
            if not torch.isfinite(loss) or not count:
                raise FloatingPointError('nonfinite validation loss or empty supervision')
            total += loss.item() * count
            tokens += count
    if not tokens:
        raise ValueError('empty validation loader')
    return total / tokens


def monitor_step(model, processor, val_loader, train_rows, val_rows, root, output,
                 step, train_loss, max_new_tokens):
    import torch
    started = time.perf_counter()
    was_training = model.training
    try:
        model.eval()
        report = {'step': step, 'train_loss': train_loss, 'val_loss': validation_loss(model, val_loader, model.device),
                  'max_new_tokens': max_new_tokens, 'do_sample': False,
                  'val_loss_samples': len(val_loader.dataset)}
        predictor = TransformersDietModel(model, processor, max_new_tokens)
        # 保持评估不积累梯度，且固定提示仅包含图片和统一任务说明。
        with torch.no_grad():
            for split, rows in [('train', train_rows), ('val', val_rows)]:
                if rows:
                    report[split] = evaluate_split(predictor, rows, root, split)
        report['monitor_seconds'] = time.perf_counter() - started
        directory = Path(output) / 'monitor'
        directory.mkdir(exist_ok=True)
        with (directory / f'step-{step:06d}.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        return report
    finally:
        model.train(was_training)


def overfit_decision(step, train_schema_rate, cap=100):
    if step >= cap and train_schema_rate == 0:
        return 'STOP_AND_DIAGNOSE: train schema validity is zero at step cap; do not add epochs'
    if train_schema_rate == 0:
        return 'REVIEW: schema not learned yet; inspect raw text, supervision audit and token limits'
    return 'REVIEW: schema learning observed; check numeric types, F1, MSE and validation before full training'
