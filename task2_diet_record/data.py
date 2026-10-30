"""只读标注与图片，数据根目录默认为 data。"""
import json
from pathlib import Path
from .schema import validate_record, parse_output


def read_annotations(path):
    records = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'{path}:{line_number}: empty line')
            try:
                record = parse_output(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: expected object')
            records.append(record)
    return records


def image_path(root, relative):
    """禁止路径逃逸，严格按 data/images/... 解析。"""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'image outside data root: {relative}')
    return path


def load_image(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        return image.convert('RGB')


class DietDataset:
    def __init__(self, annotations, data_root='data', split=None):
        self.records = read_annotations(annotations)
        self.root = Path(data_root)
        for i, record in enumerate(self.records):
            if split and record.get('split') != split:
                raise ValueError(f'row {i}: expected split {split}')
            errors = validate_record(record.get('target'))
            if errors or parse_output(record['target_text']) != record['target']:
                raise ValueError(f'row {i}: invalid or inconsistent target: {errors}')
            image_path(self.root, record['image'])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {'image': load_image(image_path(self.root, record['image'])),
                'target_text': json.dumps(record['target'], ensure_ascii=False, separators=(',', ':' ))}
