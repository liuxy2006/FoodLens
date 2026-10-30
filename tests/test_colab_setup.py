"""小型人工 ZIP 与安装计划测试，不联网、不安装依赖、不需要 GPU。"""
import io
import ast
import json
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo
from PIL import Image
from task2_diet_record.prepare_colab_data import prepare_data, zip_files
from task2_diet_record.colab_dependencies import check_cuda_unchanged, check_install_plan
from task2_diet_record.checkpoint import file_sha256


class ZipPreparationTests(unittest.TestCase):
    def setUp(self):
        Path('outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='outputs')
        self.base = Path(self.temp.name)
        self.source = self.base / 'drive/dataset'
        self.source.mkdir(parents=True)
        self.root = self.base / 'content/foodlens_data'
        self.cache = self.base / 'content/foodlens_archives'
        self.counts = dict.fromkeys(('train', 'val', 'test'), 1)
        image = io.BytesIO()
        Image.new('RGB', (3, 3), 'red').save(image, format='PNG')
        self.png = image.getvalue()

    def tearDown(self):
        self.temp.cleanup()

    def make_zips(self, ann_prefix='annotations/', img_prefix='images/', missing=None, duplicate_root=False):
        self.annotation_bytes = {}
        with ZipFile(self.source / 'annotations.zip', 'w') as archive:
            for split in self.counts:
                row = {'dish_id': split, 'split': split, 'image': f'images/{split}/food.png',
                       'target': {'unchanged': True}, 'target_text': 'keep original text'}
                raw = (json.dumps(row) + '\n').encode()
                self.annotation_bytes[split] = raw
                archive.writestr(ann_prefix + split + '.jsonl', raw)
                if duplicate_root:
                    archive.writestr('other/' + split + '.jsonl', raw)
        with ZipFile(self.source / 'images.zip', 'w') as archive:
            for split in self.counts:
                if split != missing:
                    archive.writestr(img_prefix + split + '/food.png', self.png)
                    if missing:
                        archive.writestr(img_prefix + split + '/extra.png', self.png)
            if missing:
                archive.writestr(img_prefix + missing + '/different.png', self.png)

    def prepare(self):
        return prepare_data(self.source, self.root, self.cache, self.counts)

    def test_images_top_level(self):
        self.make_zips()
        source_hashes = {p.name: file_sha256(p) for p in self.source.iterdir()}
        result = self.prepare()
        self.assertFalse(result['reused'])
        self.assertEqual(result['splits']['test']['matched_images'], 1)
        for split in self.counts:
            self.assertEqual((self.root / 'raw/annotations' / f'{split}.jsonl').read_bytes(), self.annotation_bytes[split])
            self.assertEqual((self.root / f'images/{split}/food.png').read_bytes(), self.png)
        self.assertFalse((self.root / 'images/images').exists())
        self.assertEqual(source_hashes, {p.name: file_sha256(p) for p in self.source.iterdir()})

    def test_direct_split_directories(self):
        self.make_zips(ann_prefix='', img_prefix='')
        self.assertEqual(self.prepare()['splits']['train']['annotations'], 1)
        self.assertTrue((self.root / 'images/train/food.png').is_file())

    def test_nested_wrappers(self):
        self.make_zips(ann_prefix='dataset/raw/annotations/', img_prefix='export/dataset/images/')
        self.assertEqual(self.prepare()['splits']['val']['matched_images'], 1)

    def test_reuse_without_copy_or_extraction(self):
        self.make_zips()
        self.prepare()
        before = (self.root / 'images/train/food.png').stat().st_mtime_ns
        with patch('task2_diet_record.prepare_colab_data.shutil.copyfileobj', side_effect=AssertionError('must reuse')):
            self.assertTrue(self.prepare()['reused'])
        self.assertEqual(before, (self.root / 'images/train/food.png').stat().st_mtime_ns)

    def test_changed_prepared_image_not_overwritten(self):
        self.make_zips()
        self.prepare()
        image = self.root / 'images/train/food.png'
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            self.prepare()
        self.assertEqual(image.read_bytes(), b'changed')

    def test_missing_zip(self):
        with self.assertRaisesRegex(FileNotFoundError, 'annotations.zip'):
            self.prepare()

    def test_missing_image(self):
        self.make_zips(missing='test')
        with self.assertRaisesRegex(FileNotFoundError, 'missing annotated image'):
            self.prepare()

    def test_count_mismatch(self):
        self.make_zips()
        self.counts['train'] = 2
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            self.prepare()

    def test_ambiguous_annotation_root(self):
        self.make_zips(duplicate_root=True)
        with self.assertRaisesRegex(ValueError, 'exactly one folder'):
            self.prepare()

    def test_incomplete_existing_directory(self):
        self.make_zips()
        self.root.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, 'without a completed'):
            self.prepare()

    def test_changed_source_not_overwritten(self):
        self.make_zips()
        self.prepare()
        with (self.source / 'images.zip').open('ab') as stream:
            stream.write(b'changed-source')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.prepare()

    def test_no_writes_inside_original_zip_directory(self):
        self.make_zips()
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare_data(self.source, self.source / 'extracted', self.cache, self.counts)

    def test_cache_must_not_overlap_extracted_data(self):
        self.make_zips()
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            prepare_data(self.source, self.root, self.root / 'cache', self.counts)

    def test_unsafe_zip_paths_and_symlink(self):
        for name in ('../escape.txt', '/absolute', 'C:/drive'):
            with self.subTest(name=name), io.BytesIO() as buffer:
                with ZipFile(buffer, 'w') as archive:
                    archive.writestr(name, b'bad')
                with ZipFile(buffer) as archive, self.assertRaisesRegex(ValueError, 'unsafe'):
                    zip_files(archive)
        # Windows 的 zipfile 写入夹具时会规范化反斜杠，直接提供原始清单测试该分支。
        raw_entry = types.SimpleNamespace(filename='images\\train\\food.png', external_attr=0)
        raw_archive = types.SimpleNamespace(infolist=lambda: [raw_entry])
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            zip_files(raw_archive)
        with io.BytesIO() as buffer:
            with ZipFile(buffer, 'w') as archive:
                entry = ZipInfo('images/link'); entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(entry, b'../outside')
            with ZipFile(buffer) as archive, self.assertRaisesRegex(ValueError, 'unsafe'):
                zip_files(archive)


class DependencyGuardTests(unittest.TestCase):
    def test_keep_working_cuda(self):
        state = {'version': '2.8.0+cu126', 'cuda': '12.6', 'available': True}
        check_cuda_unchanged(state, state.copy())
        for after in ({**state, 'available': False}, {**state, 'version': '2.8.0+cpu'}, {**state, 'cuda': None}):
            with self.assertRaisesRegex(RuntimeError, 'changed or became unavailable'):
                check_cuda_unchanged(state, after)

    def test_require_gpu_before_install(self):
        state = {'version': 'cpu', 'cuda': None, 'available': False}
        with self.assertRaisesRegex(RuntimeError, 'before installation'):
            check_cuda_unchanged(state, state)

    def test_reject_gpu_runtime_install_plan(self):
        for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'nvidia-cublas-cu12',
                     'intel_sycl_rt', 'onemkl-sycl-blas', 'dpcpp-cpp-rt', 'pytorch-triton-xpu'):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'GPU runtime'):
                check_install_plan({'install': [{'metadata': {'name': name}}]})

    def test_reject_windows_or_xpu_wheels(self):
        for url in ('https://example/extra-win_amd64.whl', 'https://example/extra%2Bxpu.whl'):
            with self.assertRaises(RuntimeError):
                check_install_plan({'install': [{'metadata': {'name': 'extra'}, 'download_info': {'url': url}}]})

    def test_accept_python_dependencies(self):
        check_install_plan({'install': [{'metadata': {'name': name}} for name in ('transformers', 'peft', 'accelerate')]})
        check_install_plan({'install': []})


class NotebookSetupTests(unittest.TestCase):
    def test_expected_commit_is_required_before_clone(self):
        notebook = json.loads(Path('colab/member_c_train.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'
                      and 'EXPECTED_CODE_COMMIT = ' in ''.join(c['source']))
        guard = source.split('if not REPO.exists():')[0]
        for value in ('', '6a8ac50', 'z' * 40):
            with self.subTest(value=value), self.assertRaises(ValueError):
                exec(guard.replace('EXPECTED_CODE_COMMIT = ""', f'EXPECTED_CODE_COMMIT = {value!r}'), {'Path': Path})
        namespace = {'Path': Path}
        exec(guard.replace('EXPECTED_CODE_COMMIT = ""', 'EXPECTED_CODE_COMMIT = ' + repr('A' * 40)), namespace)
        self.assertEqual(namespace['EXPECTED_CODE_COMMIT'], 'a' * 40)
        ast.parse(source)
        self.assertIn('assert CODE_COMMIT == EXPECTED_CODE_COMMIT', source)

    def test_notebook_uses_zip_preparation_and_guarded_install(self):
        notebook = json.loads(Path('colab/member_c_train.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn('DATASET_DIR = DRIVE_ROOT / "dataset"', source)
        self.assertIn('DATA_ROOT = Path("/content/foodlens_data")', source)
        self.assertIn('prepare_data(DATASET_DIR, DATA_ROOT, ARCHIVE_CACHE)', source)
        self.assertIn('run_module("task2_diet_record.colab_dependencies"', source)
        self.assertNotIn('DATA_ROOT = DRIVE_ROOT / "data"', source)
        self.assertNotIn('"pip", "install"', source)


if __name__ == '__main__':
    unittest.main()
