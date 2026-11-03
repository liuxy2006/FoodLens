"""逐张审计真实 Processor 的监督边界和终止 token，不执行训练更新。"""
import json
from .data import load_image, image_path
from .interfaces import PROMPT, user_messages, image_prompt_inputs
from .generation_trace import prompt_hash
from .schema import parse_output, validate_record
from .vlm import SupervisedCollator


def inspect_supervision(tokenizer, ids, labels, attention, inference_ids, rendered_prompt,
                        rendered_full, target_text, image_token_id, stop_ids, max_length):
    boundary = len(inference_ids)
    supervised = [token for token in labels if token != -100]
    decoded = tokenizer.decode(supervised, skip_special_tokens=True)
    try:
        matches = parse_output(decoded) == parse_output(target_text)
    except (ValueError, TypeError):
        matches = False
    suffix = rendered_full[len(rendered_prompt):] if rendered_full.startswith(rendered_prompt) else ''
    target_position = suffix.find(target_text)
    closure = suffix[target_position + len(target_text):] if target_position >= 0 else ''
    closure_ids = tokenizer.encode(closure, add_special_tokens=False) if closure else []
    closure_special = [token for token in closure_ids if token in set(tokenizer.all_special_ids)]
    shape_ok = len(ids) == len(labels) == len(attention)
    image_positions = [i for i, token in enumerate(ids) if token == image_token_id]
    checks = {
        'token_lengths_match': shape_ok,
        'prompt_is_exact_token_prefix': ids[:boundary] == inference_ids,
        'rendered_prompt_is_full_text_prefix': rendered_full.startswith(rendered_prompt),
        'supervised_json_matches_target': matches,
        'only_answer_supervised': shape_ok and bool(supervised) and all(label == -100 for label in labels[:boundary])
            and all(labels[i] == ids[i] for i in range(boundary, len(ids)) if attention[i]),
        'padding_masked': shape_ok and all(labels[i] == -100 for i, active in enumerate(attention) if not active),
        'image_tokens_present_and_masked': bool(image_positions) and shape_ok and all(labels[i] == -100 for i in image_positions),
        'not_truncated': len(ids) <= max_length and matches,
        'termination_included_and_supervised': bool(closure_ids and closure_special) and
            ids[-len(closure_ids):] == closure_ids and labels[-len(closure_ids):] == closure_ids,
        'termination_matches_generation_eos': bool(set(closure_special) & set(stop_ids))}
    return {'checks': checks, 'all_checks_passed': all(checks.values()), 'answer_start': boundary,
            'input_token_ids': ids, 'labels': labels, 'attention_mask': attention,
            'inference_prompt_ids': inference_ids, 'supervised_token_ids': supervised,
            'decoded_supervision': decoded, 'decoded_supervision_with_special': tokenizer.decode(supervised, skip_special_tokens=False),
            'rendered_prompt_tail': rendered_prompt[-500:], 'rendered_full_tail': rendered_full[-1500:],
            'termination_text': closure, 'termination_token_ids': closure_ids,
            'termination_special_ids': closure_special, 'generation_eos_ids': stop_ids,
            'image_token_count': len(image_positions), 'supervised_token_count': len(supervised),
            'masked_token_count': sum(label == -100 for label in labels)}


def audit_one(processor, row, root, image_token_id, stop_ids, max_length):
    import torch
    if row.get('split') != 'train' or validate_record(row['target']):
        raise ValueError('supervision audit requires a valid train target')
    if parse_output(row['target_text']) != row['target']:
        raise ValueError('original target_text differs from target')
    target_text = json.dumps(row['target'], ensure_ascii=False, separators=(',', ':'))
    image = load_image(image_path(root, row['image']))
    try:
        rendered, inference = image_prompt_inputs(processor, image)
        full = processor.apply_chat_template(user_messages() + [{'role': 'assistant',
            'content': [{'type': 'text', 'text': target_text}]}], add_generation_prompt=False)
        # 标签构造与训练相同，但不因 JSON 不一致提前退出，以便保存错误样本的完整 token。
        batch = SupervisedCollator(processor, max_length, verify_answers=False)([{'image': image, 'target_text': target_text}])
        report = inspect_supervision(processor.tokenizer, batch['input_ids'][0].tolist(), batch['labels'][0].tolist(),
            batch['attention_mask'][0].tolist(), inference['input_ids'][0].tolist(), rendered, full,
            target_text, image_token_id, stop_ids, max_length)
        report['pixel_values_shape'] = list(batch['pixel_values'].shape)
        report['expected_target'] = row['target']
        report['canonical_target_text'] = target_text
        report['checks']['original_target_text_matches_target'] = True
        report['checks']['training_inference_pixels_identical'] = torch.equal(batch['pixel_values'], inference['pixel_values'])
        report['all_checks_passed'] = all(report['checks'].values())
        return report
    finally:
        image.close()


def audit_training_rows(processor, rows, root, image_token_id, stop_ids, max_length=4096):
    reports = []
    for row in rows:
        try:
            result = audit_one(processor, row, root, image_token_id, stop_ids, max_length)
        except Exception as exc:
            result = {'all_checks_passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        reports.append({'dish_id': row['dish_id'], 'image': row['image'], **result})
    return {'prompt': PROMPT, 'prompt_sha256': prompt_hash(PROMPT), 'samples': len(rows),
            'passed_samples': sum(r['all_checks_passed'] for r in reports),
            'all_checks_passed': bool(reports) and all(r['all_checks_passed'] for r in reports),
            'samples_report': reports, 'scope': 'train-only processor/token audit; no optimizer or model forward'}
