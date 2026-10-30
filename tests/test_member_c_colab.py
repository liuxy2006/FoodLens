"""Colab 入口和断点的无 GPU 测试；不冒充真实模型集成结果。"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from evaluation.diet_metrics import evaluate_predictions, matched_item_weight_errors
from task2_diet_record.checkpoint import inspect_checkpoint, file_sha256, validate_resume_signature
from task2_diet_record.vlm import configure_lora, load_backbone
from task2_diet_record.schema import NUMERIC_FIELDS


def record(weight=10):
    return {'items': [{'name': 'Rice', 'weight_g': weight}], **dict.fromkeys(NUMERIC_FIELDS, weight)}


class EvaluationScopeTests(unittest.TestCase):
    def test_valid_and_all_sample_denominators(self):
        result = evaluate_predictions([json.dumps(record(12)), 'invalid'], [record(), record()])
        self.assertEqual(result['valid_only']['denominator'], 1)
        self.assertEqual(result['all_samples_with_failure_policy']['denominator'], 2)
        self.assertEqual(result['valid_only']['micro']['f1'], 1)
        self.assertEqual(result['all_samples_with_failure_policy']['macro']['f1'], 0.5)
        self.assertEqual(result['valid_only']['mse']['fat_g'], 4)
        self.assertEqual(result['all_samples_with_failure_policy']['mse_zero_imputation']['fat_g'], 52)
        self.assertEqual(result['matched_item_weight_diagnostic']['mae_g'], 2)

    def test_no_valid_predictions_are_null(self):
        result = evaluate_predictions([None], [record()])
        self.assertIsNone(result['valid_only']['micro'])
        self.assertIsNone(result['valid_only']['macro'])
        self.assertIsNone(result['matched_item_weight_diagnostic']['mse_g2'])

    def test_matched_weight_aggregation(self):
        prediction = record(3)
        prediction['items'] += [{'name': ' RICE ', 'weight_g': 4}, {'name': 'bean', 'weight_g': 50}]
        self.assertEqual(matched_item_weight_errors(prediction, record(10)), [-3])
        self.assertEqual(matched_item_weight_errors({'items': []}, record()), [])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        Path('outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='outputs')
        self.root = Path(self.temp.name)
        self.metadata = {'signature': {'model': 'revision', 'data': 'sha'}, 'optimizer_steps': 1,
                         'next_epoch': 0, 'next_sample': 1}
        for name in ('training_state.pt', 'adapter_config.json', 'adapter_model.safetensors',
                     'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (self.root / name).write_bytes(b'unit-test-placeholder')
        (self.root / 'metadata.json').write_text(json.dumps(self.metadata), encoding='utf-8')
        self.manifest = {'files': {p.name: file_sha256(p) for p in self.root.iterdir()}}

    def tearDown(self):
        self.temp.cleanup()

    def ready(self):
        (self.root / 'READY.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_complete_manifest(self):
        self.ready()
        self.assertEqual(inspect_checkpoint(self.root), self.metadata)

    def test_incomplete_write_not_resumable(self):
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint(self.root)
        self.manifest['files'].pop('tokenizer.json')
        self.ready()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            inspect_checkpoint(self.root)

    def test_corrupt_file(self):
        self.ready()
        (self.root / 'training_state.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            inspect_checkpoint(self.root)

    def test_path_escape(self):
        self.manifest['files']['../outside'] = 'not-a-hash'
        self.ready()
        with self.assertRaisesRegex(ValueError, 'integrity'):
            inspect_checkpoint(self.root)

    def test_signature_changes_rejected(self):
        validate_resume_signature({'config': 1, 'data': 'a'}, {'config': 1, 'data': 'a'})
        for current in ({'config': 2, 'data': 'a'}, {'config': 1, 'data': 'b'}, {'config': 1}):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_resume_signature({'config': 1, 'data': 'a'}, current)

    def test_hash(self):
        path = self.root / 'abc'
        path.write_bytes(b'abc')
        self.assertEqual(file_sha256(path), hashlib.sha256(b'abc').hexdigest())


class LoRARoutingTests(unittest.TestCase):
    def test_only_text_attention_and_trainable_resume(self):
        class Model:
            config = types.SimpleNamespace(use_cache=True)
            def named_modules(self):
                return [(name, object()) for name in ['model.text_model.layer.q_proj',
                        'model.text_model.layer.v_proj', 'model.vision_model.layer.q_proj']]
            def enable_input_require_grads(self):
                self.input_grad_enabled = True
            def gradient_checkpointing_enable(self, **kwargs):
                self.checkpointing = kwargs
        calls = {}
        def load(model, path, **kwargs):
            calls.update(kwargs)
            return model
        stub = types.SimpleNamespace(LoraConfig=lambda **kwargs: kwargs,
               PeftModel=types.SimpleNamespace(from_pretrained=load),
               get_peft_model=lambda model, config: model)
        config = {'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05}
        with patch.dict(sys.modules, {'peft': stub}):
            model, targets = configure_lora(Model(), config)
            self.assertEqual(len(targets), 2)
            self.assertTrue(model.input_grad_enabled)
            self.assertFalse(model.config.use_cache)
            configure_lora(Model(), config, 'checkpoint')
            self.assertTrue(calls['is_trainable'])
            self.assertTrue(calls['local_files_only'])

    def test_processor_from_adapter_and_offline_default(self):
        calls = {}
        class Model:
            def to(self, device):
                calls['device'] = device
        def processor(path, **kwargs):
            calls['processor'] = (path, kwargs)
            return types.SimpleNamespace(tokenizer=types.SimpleNamespace(padding_side='left'))
        def model(path, **kwargs):
            calls['model'] = (path, kwargs)
            return Model()
        transformers = types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=processor),
                   AutoModelForVision2Seq=types.SimpleNamespace(from_pretrained=model))
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(float32='float32'), 'transformers': transformers}):
            _, proc = load_backbone('base', processor_path='saved_adapter')
        self.assertEqual(calls['processor'][0], 'saved_adapter')
        self.assertTrue(calls['model'][1]['local_files_only'])
        self.assertEqual(proc.tokenizer.padding_side, 'right')


class NotebookTests(unittest.TestCase):
    def test_notebook_syntax_and_no_fabricated_outputs(self):
        notebook = json.loads(Path('colab/member_c_train.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        ids = []
        for index, cell in enumerate(notebook['cells']):
            ids.append(cell['id'])
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'notebook_cell_{index}')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        self.assertEqual(len(ids), len(set(ids)))
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        for expected in ['feat/member-c-diet-record', 'drive.mount', 'torch.cuda.is_available()',
                         'RUN_FULL_TRAINING = False', '--resume', 'task2_diet_record.train',
                         'evaluation.diet_metrics', '--verify-update', '--verify-reload']:
            self.assertIn(expected, source)
        self.assertNotIn('optimizer.step()', source)
        self.assertNotIn('loss.backward()', source)

    def test_train_cli_exposes_resume_without_gpu_import(self):
        result = subprocess.run([sys.executable, '-m', 'task2_diet_record.train', '--help'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'--resume', result.stdout)
        self.assertIn(b'--verify-reload', result.stdout)


if __name__ == '__main__':
    unittest.main()
