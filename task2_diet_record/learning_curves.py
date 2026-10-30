"""从真实监控 JSON 生成 CSV 和曲线，不推算不存在的实验指标。"""
import argparse
import csv
import json
from pathlib import Path
from .schema import NUMERIC_FIELDS


def load_reports(runs):
    reports = []
    fixed_ids = {}
    for run in runs:
        paths = sorted((Path(run) / 'monitor').glob('step-*.json'))
        if not paths:
            raise ValueError(f'no monitoring reports: {run}')
        for path in paths:
            report = json.loads(path.read_text(encoding='utf-8'))
            for split in ('train', 'val'):
                if split in report:
                    ids = report[split]['dish_ids']
                    if split in fixed_ids and fixed_ids[split] != ids:
                        raise ValueError('cannot compare curves with different fixed evaluation images')
                    fixed_ids[split] = ids
            reports.append(report)
    return sorted(reports, key=lambda report: report['step'])


def curve_rows(reports):
    rows = []
    for report in reports:
        row = {field: report[field] for field in ('step', 'train_loss', 'val_loss')}
        for split in ('train', 'val'):
            if split not in report:
                continue
            metrics = report[split]['metrics']
            for field in ('json_parse_rate', 'schema_valid_rate'):
                row[f'{split}_{field}'] = metrics[field]
            row[f'{split}_valid_coverage'] = metrics['valid_predictions'] / metrics['samples']
            row[f'{split}_f1_all'] = metrics['micro']['f1']
            valid = metrics['valid_only']['micro']
            row[f'{split}_f1_valid'] = valid['f1'] if valid else None
            for field in NUMERIC_FIELDS:
                row[f'{split}_{field}_mse_valid'] = metrics['mse_valid_only'][field]
                row[f'{split}_{field}_mse_zero_imputed'] = metrics['mse_all_zero_fallback'][field]
        rows.append(row)
    return rows


def render_curves(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = curve_rows(load_reports(runs))
    updates = [json.loads(line) for run in runs for line in (Path(run) / 'steps.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    updates.sort(key=lambda update: update['step'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'learning_curves.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    with (output / 'training_steps.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = list(dict.fromkeys(key for update in updates for key in update))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(updates)
    figure, axes = plt.subplots(3, 3, figsize=(17, 12))
    panels = [('Loss', ['train_loss', 'val_loss']),
              ('JSON / Schema rates', [f'{s}_{k}' for s in ('train', 'val') for k in ('json_parse_rate', 'schema_valid_rate')]),
              ('Ingredient micro F1', [f'{s}_f1_{k}' for s in ('train', 'val') for k in ('all', 'valid')]),
              ('Valid prediction coverage', [f'{s}_valid_coverage' for s in ('train', 'val')])]
    panels += [(f'{field} MSE', [f'{s}_{field}_mse_{k}' for s in ('train', 'val') for k in ('valid', 'zero_imputed')]) for field in NUMERIC_FIELDS]
    for ax, (title, fields) in zip(axes.flat, panels):
        for field in fields:
            if field in columns:
                if field == 'train_loss' and updates:
                    ax.plot([u['step'] for u in updates], [u['loss'] for u in updates], label='train_loss_per_update')
                    continue
                # 缺少合法预测的点保留为缺失，不画成零误差。
                ax.plot([r['step'] for r in rows], [r.get(field) if r.get(field) is not None else float('nan') for r in rows], marker='o', label=field)
        ax.set_title(title); ax.set_xlabel('Optimizer steps'); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    figure.suptitle('Train subset = memorization; fixed validation subset = monitoring only')
    figure.tight_layout()
    figure.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(figure)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    render_curves(args.runs, args.output)
