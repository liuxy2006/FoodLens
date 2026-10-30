"""将 Drive 的两个 ZIP 只读复制到本地，识别层级并准备统一数据根目录。"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile
from .checkpoint import file_sha256
from .data import read_annotations, image_path

COUNTS = {'train': 2338, 'val': 414, 'test': 506}
MARKER = '.foodlens_ready.json'


def zip_files(archive):
    """先检查完整清单，再提取；拒绝路径逃逸、符号链接和重复文件。"""
    files = {}
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if ('\\' in entry.filename or path.is_absolute() or '..' in path.parts
                or ':' in entry.filename or stat.S_ISLNK(entry.external_attr >> 16)):
            raise ValueError(f'unsafe ZIP entry: {entry.filename}')
        if not entry.is_dir():
            name = path.as_posix()
            if name in files:
                raise ValueError(f'duplicate ZIP entry: {name}')
            files[name] = entry
    return files


def annotation_root(files):
    candidates = {PurePosixPath(name).parent for name in files if PurePosixPath(name).name == 'train.jsonl'}
    candidates = [root for root in candidates if all((root / f'{split}.jsonl').as_posix() in files for split in COUNTS)]
    if len(candidates) != 1:
        raise ValueError('annotations.zip must contain exactly one folder with train/val/test.jsonl')
    return candidates[0]


def image_root(files):
    roots = {split: set() for split in COUNTS}
    for name in files:
        parts = PurePosixPath(name).parts
        for index, part in enumerate(parts[:-1]):
            if part in roots:
                roots[part].add(PurePosixPath(*parts[:index]))
    candidates = set.intersection(*roots.values())
    if len(candidates) != 1:
        raise ValueError('images.zip must contain exactly one common parent of train/val/test directories')
    return candidates.pop()


def validate_prepared(root, expected_counts):
    result = {}
    all_ids, all_images = set(), set()
    for split, expected in expected_counts.items():
        rows = read_annotations(root / 'raw/annotations' / f'{split}.jsonl')
        if len(rows) != expected:
            raise ValueError(f'{split}: expected {expected} annotations, found {len(rows)}')
        for row in rows:
            relative = PurePosixPath(row['image'])
            if (row.get('split') != split or relative.parts[:2] != ('images', split)
                    or '..' in relative.parts or '\\' in row['image']):
                raise ValueError(f'{split}: annotation split/image path mismatch: {row.get("image")}')
            path = image_path(root, row['image'])
            if not path.is_file():
                raise FileNotFoundError(f'missing annotated image: {path}')
            if row['dish_id'] in all_ids or path in all_images:
                raise ValueError(f'duplicate dish_id or image: {row["dish_id"]}')
            all_ids.add(row['dish_id'])
            all_images.add(path)
        result[split] = {'annotations': len(rows), 'matched_images': len(rows)}
    return result


def prepare_data(dataset_dir, destination='/content/foodlens_data', cache_dir=None, expected_counts=None):
    """仅创建新文件；已有完整结果校验后复用，已有不完整/不一致结果明确报错。"""
    dataset_dir, root = Path(dataset_dir).resolve(), Path(destination).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else root.parent / 'foodlens_archives'
    counts = COUNTS if expected_counts is None else expected_counts
    sources = {name: dataset_dir / f'{name}.zip' for name in ('annotations', 'images')}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(f'required ZIP not found: {source}')
    # 明确阻止向源 ZIP 所在目录写入缓存或解压结果。
    if root.is_relative_to(dataset_dir) or cache.is_relative_to(dataset_dir):
        raise ValueError('destination/cache must be outside the original ZIP directory')
    if cache.is_relative_to(root) or root.is_relative_to(cache):
        raise ValueError('destination and ZIP cache directories must not overlap')
    fingerprints = {name: {'sha256': file_sha256(path), 'bytes': path.stat().st_size}
                    for name, path in sources.items()}
    if root.exists():
        marker = root / MARKER
        if not marker.is_file():
            raise FileExistsError(f'{root} exists without a completed preparation marker; choose a new local directory')
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['archives'] != fingerprints or saved['expected_counts'] != counts:
            raise ValueError('prepared data does not match source ZIPs/counts; choose a new local directory')
        for relative, digest in saved['files'].items():
            path = image_path(root, relative)
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f'prepared file missing/changed: {relative}; no files overwritten')
        return {'data_root': str(root), 'reused': True, 'splits': validate_prepared(root, counts)}
    cache.mkdir(parents=True, exist_ok=True)
    cached = {}
    for name, source in sources.items():
        local = cache / source.name
        if local.exists():
            if file_sha256(local) != fingerprints[name]['sha256']:
                raise ValueError(f'cached ZIP differs/incomplete: {local}; choose a new cache directory')
        else:
            with source.open('rb') as incoming, local.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if file_sha256(local) != fingerprints[name]['sha256']:
                raise ValueError(f'ZIP copy verification failed: {local}')
        cached[name] = local
    with ZipFile(cached['annotations']) as annotations, ZipFile(cached['images']) as images:
        annotation_files, image_files = zip_files(annotations), zip_files(images)
        ann_prefix, img_prefix = annotation_root(annotation_files), image_root(image_files)
        plan = [(annotations, annotation_files[(ann_prefix / f'{s}.jsonl').as_posix()],
                 Path('raw/annotations') / f'{s}.jsonl') for s in COUNTS]
        for name, entry in image_files.items():
            path = PurePosixPath(name)
            if path.is_relative_to(img_prefix):
                relative = path.relative_to(img_prefix)
                if relative.parts and relative.parts[0] in COUNTS:
                    plan.append((images, entry, Path('images', *relative.parts)))
        if shutil.disk_usage(cache).free < sum(entry.file_size for _, entry, _ in plan):
            raise OSError('not enough local disk space to extract dataset')
        root.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for archive, entry, relative in plan:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            hashes[relative.as_posix()] = file_sha256(target)
        summary = validate_prepared(root, counts)
        (root / MARKER).write_text(json.dumps({'archives': fingerprints, 'expected_counts': counts,
                                              'files': hashes}, indent=2), encoding='utf-8')
    return {'data_root': str(root), 'reused': False, 'splits': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--destination', default='/content/foodlens_data')
    parser.add_argument('--cache-dir', default='/content/foodlens_archives')
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.dataset_dir, args.destination, args.cache_dir), indent=2))
