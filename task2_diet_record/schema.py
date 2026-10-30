"""任务二兼容 Schema：只检查结构与数值，不检查原料重量和。"""
import json
import math

NUMERIC_FIELDS = ('total_weight_g', 'total_calories_kcal', 'protein_g', 'fat_g', 'carbohydrate_g')
SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'type': 'object', 'additionalProperties': False,
    'required': ['items', *NUMERIC_FIELDS],
    'properties': {
        'items': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'weight_g'],
            'properties': {'name': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
                           'weight_g': {'type': 'number', 'minimum': 0}}}},
        **{field: {'type': 'number', 'minimum': 0} for field in NUMERIC_FIELDS}}}


def validate_record(record):
    """返回错误列表；拒绝 NaN、无穷、布尔值及额外字段。"""
    from jsonschema import Draft202012Validator
    errors = [f'{list(e.path)}: {e.message}' for e in Draft202012Validator(SCHEMA).iter_errors(record)]
    def finite(value, path=''):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f'{path}: non-finite number')
        elif isinstance(value, dict):
            for key, child in value.items():
                finite(child, f'{path}.{key}')
        elif isinstance(value, list):
            for i, child in enumerate(value):
                finite(child, f'{path}[{i}]')
    finite(record)
    return errors


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_output(text):
    """严格解析完整 JSON 或单个 Markdown JSON 围栏；不猜测、不补全答案。"""
    if not isinstance(text, str):
        raise ValueError('prediction must be a string')
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].lower() not in ('```', '```json') or lines[-1] != '```':
            raise ValueError('invalid JSON fence')
        text = '\n'.join(lines[1:-1])
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'invalid number: {value}')))


def validated_output(text):
    record = parse_output(text)
    errors = validate_record(record)
    if errors:
        raise ValueError('; '.join(errors))
    return record


if __name__ == '__main__':
    print(json.dumps(SCHEMA, ensure_ascii=False, indent=2))
