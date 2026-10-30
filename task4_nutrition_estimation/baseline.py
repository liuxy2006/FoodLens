"""仅训练集拟合非负原料营养系数；正式推理只使用模型预测的原料与重量。"""
import argparse
import json
from pathlib import Path
import numpy as np
from task2_diet_record.schema import NUMERIC_FIELDS, validate_record, validated_output
from task2_diet_record.data import read_annotations
from evaluation.diet_metrics import normalize_name


class IngredientNutritionBaseline:
    def fit(self, records):
        from scipy.optimize import nnls
        if not records or any(row.get('split') != 'train' for row in records):
            raise ValueError('baseline fitting accepts train split only')
        for row in records:
            if validate_record(row.get('target')):
                raise ValueError('invalid training target')
        self.names = sorted({normalize_name(item['name']) for row in records for item in row['target']['items']})
        self.index = {name: i for i, name in enumerate(self.names)}
        x = np.stack([self._features(row['target']['items']) for row in records])
        y = np.asarray([[row['target'][key] for key in NUMERIC_FIELDS[1:]] for row in records])
        # x 为克数；系数单位为每克的营养量，非负约束防止负营养预测。
        self.coefficients = np.column_stack([nnls(x, y[:, j])[0] for j in range(y.shape[1])])
        total_mass = x.sum()
        self.unknown_per_g = y.sum(axis=0) / total_mass if total_mass else np.zeros(4)
        return self

    def _features(self, items):
        result = np.zeros(len(self.names))
        for item in items:
            index = self.index.get(normalize_name(item['name']))
            if index is not None:
                result[index] += item['weight_g']
        return result

    def predict_items(self, items):
        # 借用结构验证检查输入，但不使用外部真实营养标签。
        errors = validate_record({'items': items, **dict.fromkeys(NUMERIC_FIELDS, 0.0)})
        if errors:
            raise ValueError(str(errors))
        unknown = sum(item['weight_g'] for item in items if normalize_name(item['name']) not in self.index)
        nutrition = self._features(items) @ self.coefficients + unknown * self.unknown_per_g
        return {'items': items, 'total_weight_g': sum(item['weight_g'] for item in items),
                **{key: float(value) for key, value in zip(NUMERIC_FIELDS[1:], nutrition)}}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'names': self.names, 'coefficients': self.coefficients.tolist(),
                                        'unknown_per_g': self.unknown_per_g.tolist(),
                                        'fit_split': 'train'}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        obj = cls(); obj.names = payload['names']; obj.index = {name: i for i, name in enumerate(obj.names)}
        obj.coefficients = np.asarray(payload['coefficients']); obj.unknown_per_g = np.asarray(payload['unknown_per_g'])
        return obj


class NutritionEstimator:
    """direct 复用生成值；ingredients 用预测原料/重量重算，均从图片开始。"""
    def __init__(self, diet_model, baseline=None):
        self.model, self.baseline = diet_model, baseline

    def predict(self, image, method='direct'):
        record = validated_output(self.model.predict(image))
        if method == 'direct':
            return record
        if method == 'ingredients' and self.baseline is not None:
            return self.baseline.predict_items(record['items'])
        raise ValueError('unknown method or missing fitted baseline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/raw/annotations/train.jsonl')
    parser.add_argument('--output', default='outputs/nutrition_baseline.json')
    args = parser.parse_args()
    model = IngredientNutritionBaseline().fit(read_annotations(args.train))
    model.save(args.output)
    print(f'Fitted train-only baseline: {len(model.names)} ingredients; saved {args.output}')
