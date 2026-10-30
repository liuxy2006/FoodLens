"""数据质量审计和独立 Schema 审计；不改写任何原始标注。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_annotations, image_path, load_image
from .schema import NUMERIC_FIELDS, parse_output, validate_record


def distribution(values):
    if not values:
        return {'count': 0}
    array = np.asarray(values, dtype=float)
    return {'count': len(values), 'min': float(array.min()), 'max': float(array.max()),
            'mean': float(array.mean()), 'std': float(array.std()),
            **{f'p{q}': float(np.percentile(array, q)) for q in (25, 50, 75, 95)}}


def schema_audit(records):
    """结构合法性独立检查，不判断重量守恒。"""
    failures = []
    for i, row in enumerate(records):
        errors = validate_record(row.get('target'))
        try:
            text_target = parse_output(row.get('target_text'))
            errors.extend(validate_record(text_target))
            if text_target != row.get('target'):
                errors.append('target_text differs from target')
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
        if errors:
            failures.append({'row': i, 'dish_id': row.get('dish_id'), 'errors': errors})
    return {'total': len(records), 'valid': len(records) - len(failures), 'failures': failures}


def audit_dataset(data_root='data'):
    root = Path(data_root)
    report = {'splits': {}, 'cross_split_overlap': {}}
    id_owners, path_owners, byte_owners, pixel_owners = (defaultdict(list) for _ in range(4))
    referenced = set()
    for split in ('train', 'val', 'test'):
        path = root / 'raw' / 'annotations' / f'{split}.jsonl'
        records = read_annotations(path)
        names, item_counts, ids, paths = Counter(), Counter(), Counter(), Counter()
        numbers = {field: [] for field in (*NUMERIC_FIELDS, 'item_weight_g', 'item_sum_minus_total_g')}
        missing, broken, issues, mass_mismatch = [], [], [], []
        loaded = 0
        for i, row in enumerate(records):
            dish_id, relative = row.get('dish_id'), row.get('image')
            ids[dish_id] += 1; paths[relative] += 1
            id_owners[dish_id].append([split, dish_id])
            path_owners[relative].append([split, dish_id])
            if row.get('split') != split:
                issues.append({'row': i, 'error': 'split mismatch'})
            try:
                image = image_path(root, relative)
                referenced.add(image)
                if not image.is_file():
                    missing.append(relative)
                else:
                    raw_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                    byte_owners[raw_hash].append([split, dish_id])
                    rgb = load_image(image)
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    pixel_owners[digest].append([split, dish_id])
                    loaded += 1
                    rgb.close()
            except (OSError, ValueError, TypeError) as exc:
                broken.append({'image': relative, 'error': str(exc)})
            target = row.get('target')
            if validate_record(target):
                continue
            item_counts[len(target['items'])] += 1
            names.update({item['name'] for item in target['items']})
            for field in NUMERIC_FIELDS:
                numbers[field].append(target[field])
            numbers['item_weight_g'].extend(item['weight_g'] for item in target['items'])
            difference = sum(item['weight_g'] for item in target['items']) - target['total_weight_g']
            numbers['item_sum_minus_total_g'].append(difference)
            if abs(difference) > 0.1:
                mass_mismatch.append({'dish_id': dish_id, 'difference_g': difference})
        report['splits'][split] = {
            'annotation_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'records': len(records), 'unique_images': len(paths), 'images_loaded': loaded,
            'missing_images': missing, 'broken_images': broken, 'quality_issues': issues,
            'duplicate_dish_ids': {key: n for key, n in ids.items() if n > 1},
            'duplicate_image_paths': {key: n for key, n in paths.items() if n > 1},
            'ingredient_occurrences_per_dish': dict(names.most_common()),
            'items_per_image': dict(sorted(item_counts.items())),
            'numeric_distributions': {key: distribution(values) for key, values in numbers.items()},
            'mass_mismatch_tolerance_g': 0.1, 'mass_mismatches': mass_mismatch,
            'schema': schema_audit(records)}
    for name, owners in [('dish_id', id_owners), ('image_path', path_owners),
                          ('file_sha256', byte_owners), ('rgb_pixel_sha256', pixel_owners)]:
        report['cross_split_overlap'][name] = [values for values in owners.values()
                                                       if len({value[0] for value in values}) > 1]
    report['duplicate_image_content'] = [values for values in byte_owners.values() if len(values) > 1]
    disk_images = {p.resolve() for p in (root / 'images').rglob('*') if p.is_file()}
    report['image_files_on_disk'] = len(disk_images)
    report['unreferenced_image_files'] = [str(p.relative_to(root.resolve())) for p in sorted(disk_images - referenced)]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='outputs/data_audit.json')
    args = parser.parse_args()
    result = audit_dataset(args.data_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'splits': {key: {field: value[field] for field in
                      ('records', 'images_loaded', 'missing_images', 'broken_images', 'duplicate_dish_ids')}
                     for key, value in result['splits'].items()},
                      'overlap_groups': {key: len(value) for key, value in result['cross_split_overlap'].items()},
                      'image_files_on_disk': result['image_files_on_disk'],
                      'unreferenced_images': len(result['unreferenced_image_files'])}, indent=2))
    print(f'Full report: {args.output}')


if __name__ == '__main__':
    main()
