"""使用临时目录和替身验证 Kaggle 入口，不联网、不安装依赖、不使用 GPU。"""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import uuid
from zipfile import ZipFile

from task2_diet_record.kaggle_setup import mounted_dataset_dir, probe_cuda, stage25_for_kaggle


NOTEBOOK = Path(__file__).resolve().parents[1] / 'kaggle/member_c_prompt_diagnostics_kaggle.ipynb'


class KagglePathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / 'kaggle/input'
        self.root.mkdir(parents=True)

    def dataset(self, relative, files=('annotations.zip', 'images.zip')):
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            (directory / name).write_bytes(b'fixture')
        return directory

    def test_auto_detect_and_explicit_path(self):
        directory = self.dataset('datasets/nested/foodlens/wrapper')
        self.assertEqual(mounted_dataset_dir(self.root), directory)
        self.assertEqual(mounted_dataset_dir(self.root, directory), directory)
        self.assertEqual(mounted_dataset_dir(self.root, directory.relative_to(self.root)), directory)

    def test_missing_mount_or_archive_has_clear_error(self):
        directory = self.dataset('partial', ('annotations.zip',))
        with self.assertRaisesRegex(FileNotFoundError, r'annotations\.zip.*images\.zip'):
            mounted_dataset_dir(self.root)
        with self.assertRaisesRegex(FileNotFoundError, r'缺少：images\.zip'):
            mounted_dataset_dir(self.root, directory)
        with self.assertRaisesRegex(FileNotFoundError, 'Add Input'):
            mounted_dataset_dir(self.root, 'not-mounted')

    def test_multiple_pairs_require_selection(self):
        first = self.dataset('version-one')
        self.dataset('version-two')
        with self.assertRaisesRegex(ValueError, 'DATASET_DIR'):
            mounted_dataset_dir(self.root)
        self.assertEqual(mounted_dataset_dir(self.root, first), first)

    def test_path_outside_kaggle_mount_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '输入目录必须在'):
            mounted_dataset_dir(self.root, '../working')

    def stage25_fixture(self, pointer):
        source = self.root / 'old-experiment/schema_step25'
        (source / 'checkpoint-000025').mkdir(parents=True)
        (source / 'monitor').mkdir()
        for relative in ('run.json', 'steps.jsonl', 'monitor/step-000025.json', 'checkpoint-000025/READY.json'):
            (source / relative).write_text('{}', encoding='utf-8')
        (source / 'last_checkpoint.json').write_text(json.dumps({'path': pointer}), encoding='utf-8')
        return source

    def test_old_colab_pointer_is_rebased_only_in_new_copy(self):
        source = self.stage25_fixture('/content/drive/MyDrive/FoodLens/outputs/schema_step25/checkpoint-000025')
        original = (source / 'last_checkpoint.json').read_bytes()
        target = self.base / 'kaggle/temp/adapter'
        self.assertEqual(stage25_for_kaggle(self.root, source, target), target)
        pointer = json.loads((target / 'last_checkpoint.json').read_text())['path']
        self.assertEqual(pointer, 'checkpoint-000025')
        self.assertTrue((target / pointer / 'READY.json').is_file())
        self.assertTrue((target / 'monitor/step-000025.json').is_file())
        self.assertEqual((source / 'last_checkpoint.json').read_bytes(), original)
        with self.assertRaises(FileExistsError):
            stage25_for_kaggle(self.root, source, target)

    def test_old_stage_pointer_does_not_accept_other_checkpoints(self):
        source = self.stage25_fixture('/content/drive/schema_step25/checkpoint-000100')
        target = self.base / 'adapter'
        with self.assertRaisesRegex(ValueError, 'checkpoint-000025'):
            stage25_for_kaggle(self.root, source, target)
        self.assertFalse(target.exists())


class CudaProbeTests(unittest.TestCase):
    def test_unavailable_cuda_stops_before_tensor_allocation(self):
        fake = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'),
                               cuda=SimpleNamespace(is_available=lambda: False), ones=Mock())
        with patch.dict(sys.modules, {'torch': fake}), self.assertRaisesRegex(RuntimeError, '不会回退到 CPU'):
            probe_cuda()
        fake.ones.assert_not_called()

    def test_actual_operation_failure_stops_even_if_cuda_is_available(self):
        tensor = MagicMock()
        tensor.__matmul__.side_effect = RuntimeError('mock kernel unavailable')
        fake = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'),
                               cuda=SimpleNamespace(is_available=lambda: True), float32=object(),
                               ones=Mock(return_value=tensor))
        with patch.dict(sys.modules, {'torch': fake}), self.assertRaisesRegex(RuntimeError, 'mock kernel unavailable'):
            probe_cuda()
        fake.ones.assert_called_once_with((2, 2), device='cuda:0', dtype=fake.float32)


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.cells = {cell['id']: ''.join(cell['source']) for cell in cls.notebook['cells']
                     if cell['cell_type'] == 'code'}

    def test_format_syntax_and_no_training_entry(self):
        self.assertEqual(self.notebook['nbformat'], 4)
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(self.cells[cell['id']], filename=cell['id'])
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        code = '\n'.join(self.cells.values())
        for name in ('task2_diet_record.train', 'task2_diet_record.schema_experiment', 'google.colab'):
            self.assertNotIn(name, code)

    def test_repo_guard_rejects_wrong_head_before_importing_code(self):
        existing = Mock()
        existing.exists.return_value = True
        commands = Mock()
        commands.check_output.side_effect = ['b' * 40, 'feat/member-c-diet-record', 'expected-origin']
        namespace = dict(SCRATCH=MagicMock(), BRANCH='feat/member-c-diet-record',
                         REPO_URL='expected-origin', EXPECTED_CODE_COMMIT='a' * 40, subprocess=commands)
        namespace['SCRATCH'].__truediv__.return_value = existing
        with self.assertRaisesRegex(RuntimeError, '代码 SHA 不匹配'):
            exec(self.cells['kaggle-repo'], namespace)
        commands.run.assert_not_called()
        self.assertNotIn('run_module', namespace)

    def test_failing_post_install_probe_blocks_download_and_diagnostics(self):
        command = Mock(side_effect=[None, None, subprocess.CalledProcessError(1, ['probe'])])
        namespace = dict(RUN_ROOT=Path('working'), subprocess=subprocess, run_module=command)
        with self.assertRaisesRegex(RuntimeError, '预检失败'):
            exec(self.cells['kaggle-deps'], namespace)
        self.assertFalse(namespace['GPU_READY'])
        for cell in ('kaggle-model', 'kaggle-run'):
            with self.assertRaisesRegex(RuntimeError, '预检'):
                exec(self.cells[cell], namespace)
        self.assertEqual(command.call_count, 3)

    def test_report_path_and_bundle_under_working(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch('builtins.print'):
            kaggle = Path(tmp)
            archives = kaggle / 'input/foodlens'
            archives.mkdir(parents=True)
            for name in ('annotations.zip', 'images.zip'):
                (archives / name).touch()
            namespace = dict(INPUT_ROOT=kaggle / 'input', WORKING=kaggle / 'working',
                             SCRATCH=kaggle / 'temp', DATASET_DIR='', CODE_COMMIT='a' * 40,
                             BRANCH='feat/member-c-diet-record', REVISION='b' * 40, os=os,
                             platform=platform, datetime=datetime, timezone=timezone, uuid=uuid,
                             json=json, shutil=shutil)
            exec(self.cells['kaggle-paths'], namespace)
            root = namespace['RUN_ROOT']
            self.assertTrue(root.is_relative_to(kaggle / 'working'))
            self.assertTrue(namespace['MODEL_DIR'].is_relative_to(kaggle / 'temp'))
            probe = namespace['DIAGNOSTICS']
            (probe / 'zero_shot').mkdir(parents=True)
            (probe / 'supervision_audit_16.json').write_text(
                json.dumps(dict(samples=16, passed_samples=16, samples_report=[])), encoding='utf-8')
            for name in ('A', 'B', 'C'):
                (probe / 'zero_shot' / f'{name}.json').write_text(
                    json.dumps(dict(prompt_sha256='fixture', summary={}, predictions=[])), encoding='utf-8')
            exec(self.cells['kaggle-results'], namespace)
            bundle = namespace['bundle']
            self.assertTrue(bundle.is_relative_to(kaggle / 'working'))
            with ZipFile(bundle) as archive:
                self.assertIn(f'{root.name}/probe/zero_shot/A.json', archive.namelist())

    def test_no_adapter_is_independent_and_uses_shared_module(self):
        namespace = dict(GPU_READY=True, MODEL_READY=True, EXISTING_STAGE25_DIR='',
                         DIAGNOSTICS=Mock(), MODEL_DIR=Path('base'), REVISION='b' * 40,
                         DATA_ROOT=Path('data'), run_module=Mock())
        namespace['DIAGNOSTICS'].exists.return_value = False
        exec(self.cells['kaggle-run'], namespace)
        module, *args = namespace['run_module'].call_args.args
        self.assertEqual(module, 'task2_diet_record.prompt_diagnostics')
        self.assertNotIn('--existing-stage25-dir', args)
        self.assertIn(512, args)


if __name__ == '__main__':
    unittest.main()
