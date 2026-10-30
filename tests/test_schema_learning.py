"""格式学习工具的 CPU 测试；人工样例不能当作模型实验成绩。"""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from task2_diet_record.interfaces import PROMPT
from task2_diet_record.vlm import verify_answer_text, SupervisedCollator
from task2_diet_record.train import validate_experiment
from task2_diet_record.schema_experiment import check_stage
from task2_diet_record.monitoring import schema_observations, evaluate_split, monitor_step, overfit_decision, validation_loss
from task2_diet_record.learning_curves import curve_rows, render_curves, load_reports
from task2_diet_record.schema import NUMERIC_FIELDS


def target():
    return {'items': [{'name': 'rice', 'weight_g': 12}], **dict.fromkeys(NUMERIC_FIELDS, 12)}


def row(split):
    return {'dish_id': split + '_fixture', 'split': split, 'image': f'images/{split}/fixture.png', 'target': target()}


class SupervisionTests(unittest.TestCase):
    def test_supervised_json_equivalence(self):
        verify_answer_text(json.dumps(target(), indent=2), json.dumps(target()))
        for decoded in ('{}', '{', '{"items":[],"protein":12}'):
            with self.subTest(decoded=decoded), self.assertRaises(ValueError):
                verify_answer_text(decoded, json.dumps(target()))

    def test_left_padding_is_rejected(self):
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(padding_side='left'),
                         apply_chat_template=lambda *args, **kwargs: 'template')
        with self.assertRaisesRegex(ValueError, 'right padding'):
            SupervisedCollator(processor)([{'image': None, 'target_text': '{}'}])

    def test_prompt_specifies_exact_nested_types(self):
        for field in ('items', 'name', 'weight_g', *NUMERIC_FIELDS):
            self.assertIn('"' + field + '"', PROMPT)
        self.assertIn('not strings', PROMPT)


class FormatMonitoringTests(unittest.TestCase):
    def test_wrong_keys_are_not_repaired(self):
        raw = json.dumps({'items': [{'ingredient_name': 'rice', 'weight_g': '12'}], 'total_weight': 12})
        details = schema_observations(raw)
        self.assertTrue(details['json_parsed'])
        self.assertFalse(details['schema_valid'])
        self.assertIn('total_weight', details['unexpected_top_level_keys'])
        self.assertIn('ingredient_name', details['unexpected_item_keys'])
        self.assertIn('protein_g', details['missing_top_level_keys'])

    def test_correct_numbers_and_invalid_boolean(self):
        details = schema_observations(json.dumps(target()))
        self.assertTrue(details['schema_valid'])
        self.assertEqual(len(details['numeric_top_level_fields']), 5)
        self.assertEqual(details['numeric_item_weight_count'], 1)
        invalid = target(); invalid['protein_g'] = True
        self.assertNotIn('protein_g', schema_observations(json.dumps(invalid))['numeric_top_level_fields'])
        self.assertFalse(schema_observations(None)['json_parsed'])

    def test_split_evaluation_only_sends_image(self):
        class Predictor:
            last_generation = {'generated_tokens': 80, 'reached_token_limit': False}
            def predict(self, image):
                self.input = image
                return json.dumps(target())
        predictor = Predictor()
        report = evaluate_split(predictor, [row('train')], 'data', 'train')
        self.assertIsInstance(predictor.input, Path)
        self.assertEqual(report['scope'], 'train_memorization_only')
        self.assertEqual(report['metrics']['schema_valid_rate'], 1)
        self.assertEqual(report['predictions'][0]['target'], target())
        self.assertEqual(report['predictions'][0]['generation']['generated_tokens'], 80)
        with self.assertRaises(ValueError):
            evaluate_split(predictor, [row('test')], 'data', 'test')

    def test_stop_rule(self):
        self.assertIn('STOP_AND_DIAGNOSE', overfit_decision(100, 0))
        self.assertIn('REVIEW', overfit_decision(25, 0))
        self.assertNotIn('STOP', overfit_decision(100, 0.5))

    def test_monitor_restores_training_mode_on_failure(self):
        class Model:
            training = True
            device = 'cpu'
            def eval(self):
                self.training = False
            def train(self, state=True):
                self.training = state
        model = Model()
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace()}), patch(
                'task2_diet_record.monitoring.validation_loss', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                monitor_step(model, None, None, [], [], 'data', 'outputs', 0, None, 10)
        self.assertTrue(model.training)

    def test_monitor_success_preserves_mode_and_writes_report(self):
        model = types.SimpleNamespace(training=True, device='cpu')
        model.eval = lambda: setattr(model, 'training', False)
        model.train = lambda state: setattr(model, 'training', state)
        predictor = types.SimpleNamespace(predict=lambda image: json.dumps(target()))
        Path('outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='outputs') as directory, patch.dict(
                sys.modules, {'torch': types.SimpleNamespace(no_grad=contextlib.nullcontext)}), patch(
                'task2_diet_record.monitoring.validation_loss', return_value=1.2), patch(
                'task2_diet_record.monitoring.TransformersDietModel', return_value=predictor):
            report = monitor_step(model, None, types.SimpleNamespace(dataset=[row('val')]),
                                  [row('train')], [row('val')], 'data', directory, 25, 1.0, 1024)
            self.assertTrue(model.training)
            self.assertEqual(report['train']['metrics']['schema_valid_rate'], 1)
            self.assertTrue((Path(directory) / 'monitor/step-000025.json').is_file())

    def test_validation_loss_uses_supervised_token_denominator(self):
        import math
        import numpy as np
        class Value:
            def __init__(self, value):
                self.value = value
            def to(self, device):
                return self
            def __getitem__(self, key):
                return np.asarray(self.value)[key]
            def item(self):
                return self.value
        model = lambda **batch: types.SimpleNamespace(loss=batch['loss_value'])
        batches = [{'labels': Value([[-100, 10, 20]]), 'loss_value': Value(1.0)},
                   {'labels': Value([[-100, 10, 20, 30, 40]]), 'loss_value': Value(3.0)}]
        torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, isfinite=lambda value: math.isfinite(value.item()))
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertAlmostEqual(validation_loss(model, batches, 'cpu'), 14/6)
            with self.assertRaises(ValueError):
                validation_loss(model, [], 'cpu')
            batches[0]['loss_value'] = Value(float('nan'))
            with self.assertRaises(FloatingPointError):
                validation_loss(model, batches, 'cpu')


class ExperimentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('task2_diet_record/overfit_config.json').read_text(encoding='utf-8'))

    def test_budget_and_cap(self):
        validate_experiment(self.config, 16, 25)
        validate_experiment(self.config, 16, 100)
        for count, steps in ((8, 25), (16, 101), (16, None)):
            with self.assertRaises(ValueError):
                validate_experiment(self.config, count, steps)
        with self.assertRaisesRegex(ValueError, 'only provide'):
            validate_experiment({**self.config, 'epochs': 1}, 16, 100)

    def test_full_experiment_is_one_epoch(self):
        config = json.loads(Path('task2_diet_record/exploratory_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['epochs'], 1)
        with self.assertRaises(ValueError):
            validate_experiment({**config, 'epochs': 3}, None, None)

    def test_fresh_25_then_reviewed_resume(self):
        check_stage(25)
        with self.assertRaisesRegex(ValueError, 'review'):
            check_stage(100)
        state = {'signature': {'config': self.config, 'train_records': 16}, 'optimizer_steps': 25}
        check_stage(100, state)
        with self.assertRaises(ValueError):
            check_stage(25, state)
        state['signature']['config'] = {'experiment_kind': 'smoke'}
        with self.assertRaisesRegex(ValueError, 'smoke'):
            check_stage(100, state)


class CurveTests(unittest.TestCase):
    def setUp(self):
        Path('outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='outputs')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture_report(self, prediction):
        predictor = types.SimpleNamespace(predict=lambda image: prediction)
        return {'step': 25, 'train_loss': 1.0, 'val_loss': 1.2,
                **{s: evaluate_split(predictor, [row(s)], self.root, s) for s in ('train', 'val')}}

    def test_no_valid_mse_is_missing_not_zero(self):
        values = curve_rows([self.fixture_report('bad')])[0]
        self.assertIsNone(values['train_protein_g_mse_valid'])
        self.assertEqual(values['train_protein_g_mse_zero_imputed'], 144)

    def test_render_actual_fixture_files(self):
        directory = self.root / 'run'
        (directory / 'monitor').mkdir(parents=True)
        (directory / 'monitor/step-000025.json').write_text(json.dumps(self.fixture_report(json.dumps(target()))), encoding='utf-8')
        (directory / 'steps.jsonl').write_text(json.dumps({'step': 25, 'loss': 1.0}) + '\n', encoding='utf-8')
        rendered = self.root / 'curves'
        render_curves([directory], rendered)
        self.assertTrue((rendered / 'learning_curves.png').read_bytes().startswith(b'\x89PNG'))
        self.assertIn('train_protein_g_mse_valid', (rendered / 'learning_curves.csv').read_text())
        self.assertTrue((rendered / 'training_steps.csv').is_file())
        with self.assertRaises(FileExistsError):
            render_curves([directory], rendered)

    def test_different_fixed_images_rejected(self):
        directory = self.root / 'run'
        (directory / 'monitor').mkdir(parents=True)
        report = self.fixture_report('{}')
        (directory / 'monitor/step-000000.json').write_text(json.dumps(report), encoding='utf-8')
        report['train']['dish_ids'] = ['different_fixture']
        (directory / 'monitor/step-000025.json').write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different fixed'):
            load_reports([directory])


class OverfitNotebookTests(unittest.TestCase):
    def test_independent_notebook_has_opt_in_limits(self):
        import ast
        notebook = json.loads(Path('colab/member_c_schema_overfit.ipynb').read_text(encoding='utf-8'))
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for source in sources:
            ast.parse(source)
        source = '\n'.join(sources)
        for required in ('CONTINUE_TO_100 = False', 'RUN_FULL_TRAINING = False',
                         'SCHEMA_LEARNING_REVIEWED = False', 'schema_overfit_', 'schema_experiment',
                         'exploratory_config.json', 'learning_curves', 'EXPECTED_CODE_COMMIT'):
            self.assertIn(required, source)
        self.assertNotIn('loss.backward()', source)
        self.assertNotIn('smoke_step1', source)
        self.assertTrue(all(cell.get('execution_count') is None and not cell.get('outputs') for cell in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
