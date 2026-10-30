"""失败预测按空原料集合、全零数值计入主指标；另报有效子集指标。"""
import argparse
import json
import re
import unicodedata
from task2_diet_record.schema import NUMERIC_FIELDS, parse_output, validate_record
from task2_diet_record.data import read_annotations


def normalize_name(name):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', name).casefold()).strip()


def ingredient_counts(prediction, target):
    pred = {normalize_name(item['name']) for item in prediction['items']}
    true = {normalize_name(item['name']) for item in target['items']}
    return len(pred & true), len(pred - true), len(true - pred)


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_predictions(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError('predictions and targets must have equal non-zero length')
    tp = fp = fn = parsed = valid = 0
    macro = dict.fromkeys(('precision', 'recall', 'f1'), 0.0)
    squared = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    valid_squared = squared.copy()
    failures = []
    for index, (text, target) in enumerate(zip(predictions, targets)):
        errors = validate_record(target)
        if errors:
            raise ValueError(f'invalid ground truth {index}: {errors}')
        prediction = None
        try:
            prediction = parse_output(text)
            parsed += 1
            errors = validate_record(prediction)
            if errors:
                raise ValueError('; '.join(errors))
            valid += 1
        except (ValueError, TypeError) as exc:
            failures.append({'index': index, 'error': str(exc)})
            prediction = {'items': [], **dict.fromkeys(NUMERIC_FIELDS, 0.0)}
        counts = ingredient_counts(prediction, target)
        tp += counts[0]; fp += counts[1]; fn += counts[2]
        for key, value in precision_recall_f1(*counts).items():
            macro[key] += value
        for field in NUMERIC_FIELDS:
            error = (prediction[field] - target[field]) ** 2
            squared[field] += error
            if not failures or failures[-1]['index'] != index:
                valid_squared[field] += error
    n = len(targets)
    return {'samples': n, 'json_parse_rate': parsed / n, 'schema_valid_rate': valid / n,
            'valid_predictions': valid, 'failed_predictions': n - valid,
            'micro': precision_recall_f1(tp, fp, fn),
            'macro': {key: value / n for key, value in macro.items()},
            'counts': {'tp': tp, 'fp': fp, 'fn': fn},
            'mse_all_zero_fallback': {key: value / n for key, value in squared.items()},
            'mse_valid_only': {key: value / valid if valid else None for key, value in valid_squared.items()},
            'failures': failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True, help='JSONL: dish_id, prediction (raw text)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = read_annotations(args.annotations)
    predictions = read_annotations(args.predictions)
    ids = [row['dish_id'] for row in rows]
    pred_ids = [row['dish_id'] for row in predictions]
    if len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids) or set(pred_ids) - set(ids):
        raise ValueError('duplicate IDs or unknown prediction IDs')
    lookup = {row['dish_id']: row.get('prediction') for row in predictions}
    result = evaluate_predictions([lookup.get(i) for i in ids], [r['target'] for r in rows])
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
