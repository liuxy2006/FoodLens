"""全部数据均为人工构造的接口测试样例，不代表模型实验结果。"""
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from evaluation.diet_metrics import (normalize_name, ingredient_counts,
                                     precision_recall_f1, evaluate_predictions)
from task2_diet_record.schema import NUMERIC_FIELDS, parse_output, validate_record, validated_output
from task2_diet_record.data import DietDataset, image_path, read_annotations
from task2_diet_record.audit import audit_dataset, schema_audit, distribution
from task2_diet_record.interfaces import CaptionDietAdapter
from task2_diet_record.vlm import answer_start
from task2_diet_record.predict import predict_records
from task4_nutrition_estimation.baseline import IngredientNutritionBaseline, NutritionEstimator


def target(name='rice', weight=10.0):
    return {'items': [{'name': name, 'weight_g': weight}],
            **{field: weight for field in NUMERIC_FIELDS}}


def annotation(split='train', name='rice', weight=10.0):
    value = target(name, weight)
    return {'dish_id': split, 'split': split, 'image': f'images/{split}/food.png',
            'target': value, 'target_text': json.dumps(value)}


class SchemaTests(unittest.TestCase):
    def test_valid_and_mass_mismatch(self):
        value = target(); value['total_weight_g'] = 100
        self.assertEqual(validate_record(value), [])
        self.assertEqual(validated_output(json.dumps(value)), value)

    def test_fence(self):
        self.assertEqual(parse_output('```json\n{}\n```'), {})

    def test_strict_parse(self):
        for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', 'text {}', '{} {}', '```json\n{}', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_output(value)

    def test_invalid_structure(self):
        cases = [None, [], {}, {**target(), 'extra': 1}, {**target(), 'protein_g': True},
                 {**target(), 'fat_g': -1}, {**target(), 'protein_g': float('nan')},
                 {**target(), 'items': [{'name': ' ', 'weight_g': 1}]},
                 {**target(), 'items': [{'name': 'rice', 'weight_g': '1'}]}]
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(validate_record(value))

    def test_empty_items_valid(self):
        self.assertFalse(validate_record({'items': [], **dict.fromkeys(NUMERIC_FIELDS, 0)}))

    def test_validated_output_rejects_schema(self):
        with self.assertRaises(ValueError):
            validated_output('{}')


class MetricTests(unittest.TestCase):
    def test_cli_alignment_and_missing_prediction(self):
        import subprocess
        import sys
        Path('outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='outputs') as directory:
            root = Path(directory)
            rows = [annotation('train'), annotation('val')]
            (root / 'truth.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            predictions = root / 'predictions.jsonl'
            predictions.write_text(json.dumps({'dish_id': 'val', 'prediction': json.dumps(target())}) + '\n', encoding='utf-8')
            command = [sys.executable, '-m', 'evaluation.diet_metrics', '--annotations', str(root / 'truth.jsonl'),
                       '--predictions', str(predictions), '--output', str(root / 'result.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((root / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['failed_predictions'], 1)
            self.assertEqual(metrics['failures'][0]['index'], 0)
            predictions.write_text(json.dumps({'dish_id': 'extra', 'prediction': '{}'}) + '\n', encoding='utf-8')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            predictions.write_text((json.dumps({'dish_id': 'val', 'prediction': '{}'}) + '\n') * 2, encoding='utf-8')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_normalize(self):
        self.assertEqual(normalize_name('  ＲＩＣＥ  \t Cake '), 'rice cake')

    def test_set_counts(self):
        pred = target(' RICE '); pred['items'].append({'name': 'rice', 'weight_g': 2})
        self.assertEqual(ingredient_counts(pred, target()), (1, 0, 0))
        self.assertEqual(ingredient_counts(target('bean'), target()), (0, 1, 1))

    def test_prf(self):
        self.assertEqual(precision_recall_f1(0, 0, 0), dict.fromkeys(('precision', 'recall', 'f1'), 0.0))
        self.assertEqual(precision_recall_f1(1, 1, 0), {'precision': 0.5, 'recall': 1.0, 'f1': 2/3})

    def test_perfect(self):
        truth = target()
        result = evaluate_predictions([json.dumps(truth)], [truth])
        self.assertEqual(result['micro']['f1'], 1)
        self.assertTrue(all(value == 0 for value in result['mse_all_zero_fallback'].values()))

    def test_failures_not_dropped(self):
        result = evaluate_predictions(['oops', '{}', json.dumps(target())], [target()] * 3)
        self.assertEqual(result['json_parse_rate'], 2/3)
        self.assertEqual(result['schema_valid_rate'], 1/3)
        self.assertEqual(result['failed_predictions'], 2)
        self.assertAlmostEqual(result['micro']['recall'], 1/3)
        self.assertAlmostEqual(result['mse_all_zero_fallback']['fat_g'], 200/3)
        self.assertEqual(result['mse_valid_only']['fat_g'], 0)
        self.assertEqual(len(result['failures']), 2)

    def test_five_mse(self):
        pred = target()
        for i, field in enumerate(NUMERIC_FIELDS, 1):
            pred[field] += i
        result = evaluate_predictions([json.dumps(pred)], [target()])
        self.assertEqual(list(result['mse_all_zero_fallback'].values()), [1, 4, 9, 16, 25])

    def test_micro_macro(self):
        t1 = target(); t1['items'] += [{'name': 'bean', 'weight_g': 1}]
        result = evaluate_predictions([json.dumps(target()), 'bad'], [t1, target()])
        self.assertEqual(result['micro']['f1'], 0.5)
        self.assertAlmostEqual(result['macro']['f1'], 1/3)

    def test_no_valid(self):
        result = evaluate_predictions([None], [target()])
        self.assertIsNone(result['mse_valid_only']['fat_g'])
        self.assertEqual(result['valid_predictions'], 0)

    def test_invalid_inputs(self):
        for predictions, targets in [([], []), ([], [target()]), (['{}'], [{}])]:
            with self.assertRaises(ValueError):
                evaluate_predictions(predictions, targets)


class DataTests(unittest.TestCase):
    def setUp(self):
        Path('outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='outputs')
        self.root = Path(self.temp.name)
        (self.root / 'raw/annotations').mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            path = self.root / f'images/{split}/food.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (4, 4), color='red').save(path)
            (self.root / f'raw/annotations/{split}.jsonl').write_text(json.dumps(annotation(split)) + '\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_dataset(self):
        data = DietDataset(self.root / 'raw/annotations/train.jsonl', self.root, 'train')
        self.assertEqual(len(data), 1)
        sample = data[0]
        self.assertEqual(sample['image'].mode, 'RGB')
        self.assertEqual(parse_output(sample['target_text']), target())
        self.assertNotIn('dish_id', sample)

    def test_path_escape(self):
        with self.assertRaises(ValueError):
            image_path(self.root, '../outside.png')

    def test_split_and_text_mismatch(self):
        path = self.root / 'raw/annotations/train.jsonl'
        with self.assertRaises(ValueError):
            DietDataset(path, self.root, 'val')
        value = annotation(); value['target_text'] = '{}'
        path.write_text(json.dumps(value) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            DietDataset(path, self.root)

    def test_bad_jsonl(self):
        path = self.root / 'bad.jsonl'
        path.write_text('[]\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_annotations(path)
        path.write_text('\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_annotations(path)

    def test_audit_leak_and_schema_separation(self):
        result = audit_dataset(self.root)
        self.assertEqual(len(result['cross_split_overlap']['file_sha256']), 1)
        self.assertEqual(len(result['cross_split_overlap']['rgb_pixel_sha256']), 1)
        value = annotation(); value['target']['total_weight_g'] = 99
        value['target_text'] = json.dumps(value['target'])
        self.assertEqual(schema_audit([value])['valid'], 1)
        self.assertEqual(distribution([1, 3])['mean'], 2)
        self.assertEqual(distribution([])['count'], 0)

    def test_audit_corrupt_and_missing(self):
        (self.root / 'images/train/food.png').write_bytes(b'not an image')
        value = annotation('val'); value['image'] = 'images/absent.png'
        (self.root / 'raw/annotations/val.jsonl').write_text(json.dumps(value) + '\n', encoding='utf-8')
        result = audit_dataset(self.root)
        self.assertEqual(len(result['splits']['train']['broken_images']), 1)
        self.assertEqual(len(result['splits']['val']['missing_images']), 1)


class InterfaceTests(unittest.TestCase):
    def test_supervision_mask_with_tensor_stub(self):
        # 无 PyTorch 环境也验证掩码逻辑；不能代替真实 Processor/模型集成测试。
        import numpy as np
        from task2_diet_record.vlm import SupervisedCollator
        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data)
            @property
            def shape(self):
                return self.data.shape
            def clone(self):
                return Tensor(self.data.copy())
            def bool(self):
                return Tensor(self.data.astype(bool))
            def tolist(self):
                return self.data.tolist()
            def __eq__(self, other):
                return self.data == other
            def __getitem__(self, key):
                return Tensor(self.data[key.data if isinstance(key, Tensor) else key])
            def __setitem__(self, key, value):
                self.data[key] = value
        class Processor:
            def apply_chat_template(self, messages, add_generation_prompt):
                return 'prompt' if add_generation_prompt else 'answer'
            def __call__(self, text, images, **kwargs):
                self.images = images
                if text[0] == 'prompt':
                    return {'input_ids': Tensor([[10, 11], [10, 11]]),
                            'attention_mask': Tensor([[1, 1], [1, 1]])}
                return {'input_ids': Tensor([[10, 11, 20, 21], [10, 11, 22, 0]]),
                        'attention_mask': Tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                        'pixel_values': Tensor([1, 2])}
        processor = Processor()
        examples = [{'image': 'image1', 'target_text': '{}'}, {'image': 'image2', 'target_text': '{}'}]
        result = SupervisedCollator(processor)(examples)
        self.assertEqual(result['labels'].tolist(), [[-100, -100, 20, 21], [-100, -100, 22, -100]])
        self.assertEqual(processor.images, [['image1'], ['image2']])
        self.assertIn('pixel_values', result)
        with self.assertRaises(ValueError):
            SupervisedCollator(processor, max_length=2)(examples)

    def test_caption_adapter(self):
        class Caption:
            def caption(self, image):
                self.image = image
                return 'rice'
        caption = Caption()
        adapter = CaptionDietAdapter(caption, lambda text: json.dumps(target(text)))
        self.assertEqual(validated_output(adapter.predict('food.png')), target())
        self.assertEqual(caption.image, Path('food.png'))

    def test_no_target_leak_and_failures(self):
        class Model:
            def predict(self, image):
                if image.name == 'bad.png':
                    raise RuntimeError('test failure')
                return json.dumps(target())
        rows = [{'dish_id': 'a', 'image': 'ok.png', 'target': 'do not send'},
                {'dish_id': 'b', 'image': 'bad.png'}]
        result = predict_records(Model(), rows, 'data')
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1]['prediction'])
        self.assertIn('RuntimeError', result[1]['error'])

    def test_answer_prefix(self):
        self.assertEqual(answer_start([1, 2, 3, 4], [1, 2]), 2)
        for full, prefix in [([1, 2], [1, 2]), ([1, 3, 4], [1, 2])]:
            with self.assertRaises(ValueError):
                answer_start(full, prefix)


class NutritionTests(unittest.TestCase):
    def test_fit_predict_save(self):
        model = IngredientNutritionBaseline().fit([annotation(weight=10), annotation(weight=20)])
        result = model.predict_items([{'name': 'RICE', 'weight_g': 5}])
        self.assertAlmostEqual(result['fat_g'], 5)
        self.assertEqual(result['total_weight_g'], 5)
        self.assertAlmostEqual(model.predict_items([{'name': 'unknown', 'weight_g': 3}])['protein_g'], 3)
        Path('outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='outputs') as directory:
            path = Path(directory) / 'baseline.json'
            model.save(path)
            self.assertEqual(IngredientNutritionBaseline.load(path).predict_items(result['items']), result)

    def test_only_train(self):
        for rows in [[], [annotation('val')], [annotation('test')]]:
            with self.assertRaises(ValueError):
                IngredientNutritionBaseline().fit(rows)

    def test_invalid_items(self):
        model = IngredientNutritionBaseline().fit([annotation()])
        with self.assertRaises(ValueError):
            model.predict_items([{'name': 'rice', 'weight_g': -1}])

    def test_unified_nutrition(self):
        class Model:
            def predict(self, image):
                return json.dumps(target())
        estimator = NutritionEstimator(Model(), IngredientNutritionBaseline().fit([annotation()]))
        self.assertEqual(estimator.predict('image.png'), target())
        self.assertAlmostEqual(estimator.predict('image.png', 'ingredients')['protein_g'], 10)
        with self.assertRaises(ValueError):
            estimator.predict('image.png', 'other')


if __name__ == '__main__':
    unittest.main()
