"""只观察原始 token，不删除复述、不补全 JSON、不替换模型答案。"""
import hashlib
from difflib import SequenceMatcher
import re


def prompt_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def eos_ids(model):
    generation = getattr(model, 'generation_config', None)
    value = getattr(generation, 'eos_token_id', None)
    if generation is None:
        value = getattr(model.config, 'eos_token_id', None)
    # generation_config 存在但 eos_token_id 为 None 时，不猜测其他 EOS 已被采用。
    return [] if value is None else [value] if isinstance(value, int) else list(value)


def check_generation_boundary(input_ids, output_ids):
    if len(output_ids) < len(input_ids) or output_ids[:len(input_ids)] != input_ids:
        raise ValueError('generate() returned a sequence without the exact input prefix; refusing to guess output slicing')
    return output_ids[len(input_ids):]


def echo_observation(text, prompt):
    """预先定义的连续词重叠诊断，不作为原料/营养评价指标。"""
    words = re.findall(r'\S+', text.casefold())
    prompt_words = re.findall(r'\S+', prompt.casefold())
    match = SequenceMatcher(None, words, prompt_words, autojunk=False).find_longest_match()
    return {'longest_matching_word_run': match.size,
            'matching_word_fraction_of_output': match.size / len(words) if words else 0.0,
            'echo_flag': match.size >= 8 and match.size / len(words) >= 0.5 if words else False,
            'rule': '>=8 consecutive whitespace words shared with prompt and >=50% of output words'}


def trace_generation(tokenizer, input_ids, output_ids, rendered_prompt, prompt, stop_ids):
    matches = len(output_ids) >= len(input_ids) and output_ids[:len(input_ids)] == input_ids
    new_ids = output_ids[len(input_ids):] if matches else []
    special_ids = set(getattr(tokenizer, 'all_special_ids', []))
    return {'prompt': prompt, 'prompt_sha256': prompt_hash(prompt),
            'rendered_prompt': rendered_prompt, 'rendered_prompt_tail': rendered_prompt[-500:],
            'input_tail_with_special_tokens': tokenizer.decode(input_ids[-160:], skip_special_tokens=False),
            'input_token_ids': input_ids, 'full_output_token_ids': output_ids,
            'new_token_ids': new_ids if matches else None,
            'input_tokens': len(input_ids), 'full_output_tokens': len(output_ids),
            'input_prefix_matches': matches,
            'full_output_with_special_tokens': tokenizer.decode(output_ids, skip_special_tokens=False),
            'new_output_with_special_tokens': tokenizer.decode(new_ids, skip_special_tokens=False) if matches else None,
            'generated_special_tokens': [{'offset': i, 'id': token,
                'token': tokenizer.convert_ids_to_tokens(token)} for i, token in enumerate(new_ids) if token in special_ids],
            'effective_eos_ids': stop_ids, 'last_generated_token': new_ids[-1] if new_ids else None,
            'ended_with_eos': bool(new_ids and new_ids[-1] in stop_ids)}
