"""Colab：16 图监督审计 + 固定 2 train/2 val 的 A/B/C 零样本对照；不训练。"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import time
from .checkpoint import file_sha256
from .data import read_annotations, image_path
from .interfaces import PROMPT, TransformersDietModel
from .generation_trace import prompt_hash, echo_observation, eos_ids
from .supervision_audit import audit_training_rows
from .monitoring import schema_observations
from evaluation.diet_metrics import evaluate_predictions

SHORT_PROMPT = ('Return food nutrition as JSON: items (array of name and weight_g), total_weight_g, '
                'total_calories_kcal, protein_g, fat_g, carbohydrate_g. Use numeric amounts.')
EXAMPLE = {'items': [{'name': 'example ingredient', 'weight_g': 1.0}], 'total_weight_g': 1.0,
           'total_calories_kcal': 1.0, 'protein_g': 0.0, 'fat_g': 0.0, 'carbohydrate_g': 0.0}
PROMPTS = {'A': PROMPT, 'B': SHORT_PROMPT, 'C': SHORT_PROMPT +
           '\nFormat example only, not the answer or measurements for this image: ' + json.dumps(EXAMPLE, separators=(',', ':'))}


def compare_prompts(model, processor, rows, root, output, variants, max_new_tokens, cohort):
    if not rows or any(row['split'] not in ('train', 'val') for row in rows):
        raise ValueError('diagnostic comparison requires fixed train/val images and never test')
    if cohort != 'pretrained_no_adapter' and variants != {'A': PROMPT}:
        raise ValueError('a trained A checkpoint may only be diagnosed with its original A prompt; do not mix B/C')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'cohort': cohort, 'max_new_tokens': max_new_tokens, 'do_sample': False,
                'prompts': {key: {'text': text, 'sha256': prompt_hash(text)} for key, text in variants.items()},
                'fixed_images': [{'dish_id': r['dish_id'], 'split': r['split'], 'image': r['image']} for r in rows]}
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    summaries = {}
    for variant, prompt in variants.items():
        predictor = TransformersDietModel(model, processor, max_new_tokens, prompt=prompt, capture_trace=True)
        results = []
        for row in rows:
            started = time.perf_counter()
            try:
                text = predictor.predict(image_path(root, row['image']))
                result = {'prediction': text, 'trace': predictor.last_trace,
                          'generation': predictor.last_generation, 'echo': echo_observation(text, prompt),
                          'schema': schema_observations(text), 'copied_format_example': parse_example(text)}
            except Exception as exc:
                result = {'prediction': None, 'trace': predictor.last_trace, 'error': f'{type(exc).__name__}: {exc}'}
            results.append({'dish_id': row['dish_id'], 'split': row['split'], 'seconds': time.perf_counter() - started, **result})
        summary = {}
        for split in ('train', 'val'):
            pairs = [(r, p) for r, p in zip(rows, results) if r['split'] == split]
            if pairs:
                summary[split] = {'metrics': evaluate_predictions([p['prediction'] for _, p in pairs], [r['target'] for r, _ in pairs]),
                    'echo_flagged': sum(p.get('echo', {}).get('echo_flag', False) for _, p in pairs),
                    'samples': len(pairs), 'scope': 'fixed-image diagnostic, not full validation'}
        summaries[variant] = summary
        (directory / f'{variant}.json').write_text(json.dumps({'prompt': prompt, 'prompt_sha256': prompt_hash(prompt),
            'summary': summary, 'predictions': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return summaries


def parse_example(text):
    from .schema import parse_output
    try:
        return parse_output(text) == EXAMPLE
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--existing-stage25-dir', help='可选：同提示词 A 的旧 adapter 诊断，不对该 adapter 跑 B/C')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.revision):
        parser.error('--revision must be the full pinned 40-character model commit SHA')
    output, root = Path(args.output), Path(args.data_root)
    if output.exists():
        raise FileExistsError('choose a new diagnostic directory; old experiment results are read-only')
    if args.max_new_tokens <= 0 or (Path(args.model_path) / 'adapter_config.json').exists():
        parser.error('use a local ORIGINAL pretrained model and positive token limit for zero-shot A/B/C')
    selected = None
    if args.existing_stage25_dir:
        from .schema_experiment import completed_stage25
        selected = completed_stage25(args.existing_stage25_dir, args.model_path, args.revision, root)
        if output.resolve().is_relative_to(Path(selected['experiment_dir'])):
            raise ValueError('diagnostic output must be outside the old experiment directory; it is read-only')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('run this entry on Colab NVIDIA GPU; no local CPU model inference fallback')
    from .vlm import load_backbone
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, processor = load_backbone(args.model_path, 'cuda', args.revision)
    train = read_annotations(root / 'raw/annotations/train.jsonl')[:16]
    val = read_annotations(root / 'raw/annotations/val.jsonl')[:2]
    if len(train) != 16 or len(val) != 2 or any(r['split'] != 'train' for r in train) or any(r['split'] != 'val' for r in val):
        raise ValueError('requires 16 train audit samples and 2 val generation samples; never reads test')
    output.mkdir(parents=True, exist_ok=False)
    template = getattr(processor, 'chat_template', None) or processor.tokenizer.chat_template
    import importlib.metadata
    (output / 'run.json').write_text(json.dumps({'model_revision': args.revision, 'code_commit':
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'chat_template': template,
        'chat_template_sha256': prompt_hash(json.dumps(template, sort_keys=True)), 'seed': 42,
        'annotation_sha256': {s: file_sha256(root / f'raw/annotations/{s}.jsonl') for s in ('train', 'val')},
        'checkpoint_cohorts': {'zero_shot': None, 'trained_A_only': selected},
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft')},
        'model_config_sha256': file_sha256(Path(args.model_path) / 'config.json'),
        'model_class': type(model).__name__, 'processor_class': type(processor).__name__}, indent=2), encoding='utf-8')
    audit = audit_training_rows(processor, train, root, model.config.image_token_id, eos_ids(model))
    (output / 'supervision_audit_16.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    fixed_rows = train[:2] + val
    summary = compare_prompts(model, processor, fixed_rows, root, output / 'zero_shot', PROMPTS, args.max_new_tokens, 'pretrained_no_adapter')
    if selected:
        from transformers import AutoProcessor
        saved_processor = AutoProcessor.from_pretrained(selected['checkpoint'], local_files_only=True)
        saved_processor.tokenizer.padding_side = 'right'
        saved_template = getattr(saved_processor, 'chat_template', None) or saved_processor.tokenizer.chat_template
        if saved_template != template:
            raise ValueError('base/adapter chat templates differ; recorded base diagnostics, do not silently replace a template')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, selected['checkpoint'], local_files_only=True)
        adapter_audit = audit_training_rows(saved_processor, train, root, model.config.image_token_id, eos_ids(model))
        (output / 'adapter_supervision_audit_16.json').write_text(json.dumps(adapter_audit, ensure_ascii=False, indent=2), encoding='utf-8')
        compare_prompts(model, saved_processor, fixed_rows, root, output / 'trained_A_only', {'A': PROMPT}, args.max_new_tokens, selected['checkpoint'])
    print(json.dumps({'supervision_passed': audit['passed_samples'], 'samples': 16, 'zero_shot_summary': summary,
                      'training_performed': False, 'report_directory': str(output)}, indent=2))


if __name__ == '__main__':
    main()
