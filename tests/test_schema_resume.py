"""已完成 25 步实验的只读复用测试；不加载模型、不需要 GPU。"""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from task2_diet_record.checkpoint import file_sha256
from task2_diet_record.interfaces import PROMPT
from task2_diet_record.schema_experiment import completed_stage25, main


class CompletedStage25Tests(unittest.TestCase):
    def setUp(self):
        Path('outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='outputs')
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / 'previous_session/schema_step25'
        self.old.mkdir(parents=True)
        self.checkpoint = self.old / 'checkpoint-000025'
        self.checkpoint.mkdir()
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"fixture":true}', encoding='utf-8')
        self.data = self.root / 'data'
        annotations = self.data / 'raw/annotations'
        annotations.mkdir(parents=True)
        for split in ('train', 'val'):
            (annotations / f'{split}.jsonl').write_text('unit test signature fixture\n', encoding='utf-8')
        self.revision = 'a' * 40
        self.signature = {
            'config': json.loads(Path('task2_diet_record/overfit_config.json').read_text(encoding='utf-8')),
            'annotation_sha256': {s: file_sha256(annotations / f'{s}.jsonl') for s in ('train', 'val')},
            'train_records': 16, 'val_limit': 4, 'model_revision': self.revision,
            'model_config_sha256': file_sha256(self.model / 'config.json'),
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'monitor_protocol': {'generate_train': 16, 'generate_val': 4, 'eval_every': 25}}
        self.metadata = {'signature': copy.deepcopy(self.signature), 'optimizer_steps': 25}
        for name in ('training_state.pt', 'adapter_model.safetensors', 'adapter_config.json',
                     'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (self.checkpoint / name).write_bytes(b'unit test placeholder, not a model')
        self.write_checkpoint()
        (self.old / 'run.json').write_text(json.dumps({'signature': self.signature}), encoding='utf-8')
        (self.old / 'steps.jsonl').write_text('{"step":25}\n', encoding='utf-8')
        (self.old / 'monitor').mkdir()
        (self.old / 'monitor/step-000025.json').write_text('{"step":25}', encoding='utf-8')
        (self.old / 'last_checkpoint.json').write_text(json.dumps({'path': str(self.checkpoint)}), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def write_checkpoint(self):
        (self.checkpoint / 'metadata.json').write_text(json.dumps(self.metadata), encoding='utf-8')
        manifest = {p.name: file_sha256(p) for p in self.checkpoint.iterdir() if p.name != 'READY.json'}
        (self.checkpoint / 'READY.json').write_text(json.dumps({'files': manifest}), encoding='utf-8')

    def resolve(self):
        return completed_stage25(self.old, self.model, self.revision, self.data)

    def hashes(self):
        return {str(p.relative_to(self.old)): file_sha256(p) for p in self.old.rglob('*') if p.is_file()}

    def test_valid_old_run_is_read_only(self):
        before = self.hashes()
        selected = self.resolve()
        self.assertEqual(selected['checkpoint'], str(self.checkpoint))
        self.assertEqual(selected['optimizer_steps'], 25)
        self.assertEqual(self.hashes(), before)

    def test_relative_checkpoint_pointer(self):
        (self.old / 'last_checkpoint.json').write_text('{"path":"checkpoint-000025"}', encoding='utf-8')
        self.assertEqual(self.resolve()['checkpoint'], str(self.checkpoint))

    def test_corrupt_checkpoint_is_rejected(self):
        (self.checkpoint / 'training_state.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.resolve()

    def test_partial_or_already_100_run_is_rejected(self):
        for step in (16, 100):
            self.metadata['optimizer_steps'] = step
            self.write_checkpoint()
            with self.assertRaisesRegex(ValueError, 'exactly 25'):
                self.resolve()

    def test_config_and_prompt_mismatches(self):
        self.metadata['signature']['config']['learning_rate'] = 0.002
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, 'config'):
            self.resolve()
        self.metadata['signature'] = copy.deepcopy(self.signature)
        self.write_checkpoint()
        with patch('task2_diet_record.schema_experiment.PROMPT', 'different prompt'):
            with self.assertRaisesRegex(ValueError, 'prompt_sha256'):
                self.resolve()

    def test_current_data_and_model_versions_are_checked(self):
        with self.assertRaisesRegex(ValueError, 'model_revision'):
            completed_stage25(self.old, self.model, 'b' * 40, self.data)
        (self.model / 'config.json').write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'model_config_sha256'):
            self.resolve()
        (self.model / 'config.json').write_text('{"fixture":true}', encoding='utf-8')
        (self.data / 'raw/annotations/train.jsonl').write_text('changed dataset', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'annotation_sha256'):
            self.resolve()

    def test_pointer_cannot_escape_selected_experiment(self):
        (self.old / 'last_checkpoint.json').write_text(json.dumps({'path': str(self.root / 'other')}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'inside the selected'):
            self.resolve()

    def test_review_artifacts_required(self):
        with self.assertRaises(FileNotFoundError):
            completed_stage25(self.root / 'nonexistent', self.model, self.revision, self.data)
        (self.old / 'monitor/step-000025.json').write_text('{"step":24}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'review report'):
            self.resolve()

    def test_cli_refuses_old_output_before_loading_torch(self):
        argv = ['schema_experiment', '--model-path', str(self.model), '--revision', self.revision,
                '--data-root', str(self.data), '--existing-stage25-dir', str(self.old),
                '--stage', '100', '--output', str(self.old)]
        before = self.hashes()
        with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
            main()
        self.assertEqual(self.hashes(), before)

    def test_cli_continues_to_total_100_with_old_checkpoint(self):
        output = self.root / 'new_session/schema_step100'
        argv = ['schema_experiment', '--model-path', str(self.model), '--revision', self.revision,
                '--data-root', str(self.data), '--existing-stage25-dir', str(self.old),
                '--stage', '100', '--output', str(output)]
        def finish_fixture_training(command, **kwargs):
            (output / 'monitor').mkdir(parents=True)
            (output / 'monitor/step-000100.json').write_text(
                json.dumps({'train': {'metrics': {'schema_valid_rate': 0}}}), encoding='utf-8')
        before = self.hashes()
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {
                'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))}), patch(
                'task2_diet_record.schema_experiment.subprocess.run', side_effect=finish_fixture_training) as run, contextlib.redirect_stdout(io.StringIO()):
            main()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--max-steps') + 1], '100')
        self.assertEqual(command[command.index('--resume') + 1], str(self.checkpoint))
        self.assertNotIn('--evaluate-start', command)
        self.assertEqual(self.hashes(), before)

    def stage_cell(self):
        notebook = json.loads(Path('colab/member_c_schema_overfit.ipynb').read_text(encoding='utf-8'))
        return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'
                    and 'EXISTING_STAGE25_DIR = ' in ''.join(c['source']))

    def notebook_namespace(self):
        return {'Path': Path, 'MODEL_DIR': self.model, 'REVISION': self.revision,
                'DATA_ROOT': self.data, 'RUN_ROOT': self.root / 'new_session', 'run_module': Mock()}

    def test_notebook_old_run_skips_training(self):
        source = self.stage_cell().replace('EXISTING_STAGE25_DIR = ""', 'EXISTING_STAGE25_DIR = ' + repr(str(self.old)))
        namespace = self.notebook_namespace()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        namespace['run_module'].assert_not_called()
        self.assertEqual(namespace['STAGE25'], self.old)

    def test_notebook_default_still_starts_new_25_step_run(self):
        namespace = self.notebook_namespace()
        with patch('task2_diet_record.schema_experiment.completed_stage25'), contextlib.redirect_stdout(io.StringIO()):
            exec(self.stage_cell(), namespace)
        call = namespace['run_module'].call_args.args
        self.assertEqual(call[call.index('--stage') + 1], 25)
        self.assertEqual(call[call.index('--output') + 1], namespace['RUN_ROOT'] / 'schema_step25')
        self.assertNotIn('--resume', call)

    def test_notebook_rejects_two_resume_modes(self):
        source = self.stage_cell().replace('EXISTING_STAGE25_DIR = ""', 'EXISTING_STAGE25_DIR = ' + repr(str(self.old)))
        source = source.replace('OVERFIT_RESUME = ""', 'OVERFIT_RESUME = "partial_checkpoint"')
        namespace = self.notebook_namespace()
        with self.assertRaisesRegex(ValueError, '不能同时指定'):
            exec(source, namespace)
        namespace['run_module'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
