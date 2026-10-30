"""生成与评价分离：预测循环只取 image，不向模型传 target。"""
import argparse
import json
from pathlib import Path
from .data import read_annotations, image_path
from .interfaces import TransformersDietModel


def predict_records(model, records, data_root, include_generation_info=False):
    results = []
    for row in records:
        try:
            text = model.predict(image_path(data_root, row['image']))
            result = {'dish_id': row['dish_id'], 'prediction': text}
            if include_generation_info:
                result['generation'] = getattr(model, 'last_generation', {}).copy()
            results.append(result)
        except Exception as exc:
            # 不漏掉失败样本；错误进入日志，评价器将 None 计入失败。
            results.append({'dish_id': row['dish_id'], 'prediction': None,
                            'error': f'{type(exc).__name__}: {exc}'})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'xpu'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, help='仅用于小样本集成测试')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    model = TransformersDietModel.from_pretrained(args.model_path, args.adapter, args.device)
    rows = read_annotations(args.annotations)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error('--limit must be positive')
        rows = rows[:args.limit]
    model.max_new_tokens = args.max_new_tokens
    predictions = predict_records(model, rows, args.data_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved {len(predictions)} predictions, including failures: {args.output}')


if __name__ == '__main__':
    main()
