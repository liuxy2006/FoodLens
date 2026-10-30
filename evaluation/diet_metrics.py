"""分别报告合法预测指标与预先约定的失败补值指标。"""
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


def matched_item_weight_errors(prediction, target):
    """仅匹配到的原料重量诊断；重名重量相加，不替代原料 F1/五项 MSE。"""
    def weights(record):
        result = {}
        for item in record['items']:
            name = normalize_name(item['name'])
            result[name] = result.get(name, 0.0) + item['weight_g']
        return result
    predicted, true = weights(prediction), weights(target)
    return [predicted[name] - true[name] for name in sorted(predicted.keys() & true.keys())]


def evaluate_predictions(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError('predictions and targets must have equal non-zero length')
    tp = fp = fn = parsed = valid = 0
    macro = dict.fromkeys(('precision', 'recall', 'f1'), 0.0)
    squared = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    valid_squared = squared.copy()
    valid_counts = [0, 0, 0]
    valid_macro = macro.copy()
    weight_errors = []
    failures = []
    for index, (text, target) in enumerate(zip(predictions, targets)):
        errors = validate_record(target)
        if errors:
            raise ValueError(f'invalid ground truth {index}: {errors}')
        prediction = None
        is_valid = False
        try:
            prediction = parse_output(text)
            parsed += 1
            errors = validate_record(prediction)
            if errors:
                raise ValueError('; '.join(errors))
            valid += 1
            is_valid = True
        except (ValueError, TypeError) as exc:
            failures.append({'index': index, 'error': str(exc)})
            prediction = {'items': [], **dict.fromkeys(NUMERIC_FIELDS, 0.0)}
        counts = ingredient_counts(prediction, target)
        tp += counts[0]; fp += counts[1]; fn += counts[2]
        for key, value in precision_recall_f1(*counts).items():
            macro[key] += value
            if is_valid:
                valid_macro[key] += value
        if is_valid:
            valid_counts = [a + b for a, b in zip(valid_counts, counts)]
            weight_errors.extend(matched_item_weight_errors(prediction, target))
        for field in NUMERIC_FIELDS:
            error = (prediction[field] - target[field]) ** 2
            squared[field] += error
            if is_valid:
                valid_squared[field] += error
    n = len(targets)
    return {'samples': n, 'json_parse_rate': parsed / n, 'schema_valid_rate': valid / n,
            'valid_predictions': valid, 'failed_predictions': n - valid,
            'micro': precision_recall_f1(tp, fp, fn),
            'macro': {key: value / n for key, value in macro.items()},
            'counts': {'tp': tp, 'fp': fp, 'fn': fn},
            'mse_all_zero_fallback': {key: value / n for key, value in squared.items()},
            'mse_valid_only': {key: value / valid if valid else None for key, value in valid_squared.items()},
            'valid_only': {
                'denominator': valid,
                'micro': precision_recall_f1(*valid_counts) if valid else None,
                'macro': {key: value / valid for key, value in valid_macro.items()} if valid else None,
                'mse': {key: value / valid if valid else None for key, value in valid_squared.items()}},
            'all_samples_with_failure_policy': {
                'denominator': n, 'policy': 'invalid/missing prediction -> empty items and zero numeric values',
                'micro': precision_recall_f1(tp, fp, fn),
                'macro': {key: value / n for key, value in macro.items()},
                'mse_zero_imputation': {key: value / n for key, value in squared.items()}},
            'metric_definitions': {
                'rate_denominator': n,
                'valid_only_denominator': valid,
                'micro': 'pooled TP/FP/FN; macro: arithmetic mean over selected images',
                'legacy_fields': 'micro/macro use all samples; mse_all_zero_fallback is imputed, not unconditional nutrition MSE'},
            'matched_item_weight_diagnostic': {
                'matched_ingredient_pairs': len(weight_errors),
                'mae_g': sum(abs(e) for e in weight_errors) / len(weight_errors) if weight_errors else None,
                'mse_g2': sum(e * e for e in weight_errors) / len(weight_errors) if weight_errors else None,
                'scope': 'schema-valid predictions and matched normalized names only; unmatched items excluded'},
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
