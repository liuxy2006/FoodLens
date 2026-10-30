"""仅在 Colab NVIDIA GPU 执行的真实推理与保存重载检查。"""
import argparse
import json
from pathlib import Path
import time
from .data import read_annotations, image_path
from .interfaces import TransformersDietModel
from .schema import parse_output, validate_record


def describe_generation(model, processor, image, max_new_tokens=256):
    import psutil
    import torch
    device = model.device.type
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictor = TransformersDietModel(model, processor, max_new_tokens)
    text = predictor.predict(image)
    if device == 'cuda':
        torch.cuda.synchronize()
    result = {'raw_text': text, 'seconds': time.perf_counter() - started,
              'max_new_tokens': max_new_tokens, 'json_parsed': False, 'schema_valid': False,
              'process_rss_bytes': psutil.Process().memory_info().rss}
    result.update(predictor.last_generation)
    try:
        record = parse_output(text)
        result['json_parsed'] = True
        result['schema_errors'] = validate_record(record)
        result['schema_valid'] = not result['schema_errors']
    except (ValueError, TypeError) as exc:
        result['parse_error'] = str(exc)
    if device == 'cuda':
        result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def verify_adapter_reload(model, processor, base_path, adapter, row, root, device, max_new_tokens=256):
    import torch
    image = image_path(root, row['image'])
    # 加载新模型会初始化随机参数；隔离 RNG，避免影响后续训练/恢复的一致性。
    devices = list(range(torch.cuda.device_count())) if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        before = describe_generation(model, processor, image, max_new_tokens)
        loaded = TransformersDietModel.from_pretrained(base_path, str(adapter), device)
        after = describe_generation(loaded.model, loaded.processor, image, max_new_tokens)
        del loaded
    if device == 'cuda':
        torch.cuda.empty_cache()
    return {'dish_id': row['dish_id'], 'before_save_reload': before, 'after_reload': after,
            'identical_text': before['raw_text'] == after['raw_text']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This integration entry requires a Colab NVIDIA GPU; no CPU fallback')
    if Path(args.output).exists():
        raise FileExistsError('choose a new report path')
    from .vlm import load_backbone
    model, processor = load_backbone(args.model_path, 'cuda')
    row = read_annotations(Path(args.data_root) / 'raw/annotations/val.jsonl')[0]
    result = describe_generation(model, processor, image_path(args.data_root, row['image']), args.max_new_tokens)
    result.update(dish_id=row['dish_id'], model_parameters=sum(p.numel() for p in model.parameters()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
