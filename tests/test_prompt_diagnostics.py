"""人工 token/模型替身测试；不下载权重，不产生真实模型实验成绩。"""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from task2_diet_record.generation_trace import check_generation_boundary, trace_generation, echo_observation, prompt_hash, eos_ids
from task2_diet_record.interfaces import PROMPT, TransformersDietModel
from task2_diet_record.supervision_audit import inspect_supervision, audit_training_rows
from task2_diet_record.prompt_diagnostics import PROMPTS, EXAMPLE, compare_prompts
from task2_diet_record.prompt_diagnostics import main as diagnostic_main


class CharacterTokenizer:
    all_special_ids = [1, 2, 3, 9]
    def encode(self, text, add_special_tokens=False):
        if '<eot>' in text:
            before, after = text.split('<eot>')
            return self.encode(before) + [9] + self.encode(after)
        return [ord(char) + 100 for char in text]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(('' if skip_special_tokens else self.convert_ids_to_tokens(token))
                       if token in self.all_special_ids else chr(token - 100) for token in ids)
    def convert_ids_to_tokens(self, token):
        return {1: '<bos>', 2: '<image>', 3: '<assistant>', 9: '<eot>'}.get(token, str(token))


class GenerationBoundaryTests(unittest.TestCase):
    def test_eos_is_not_guessed_from_other_token_sources(self):
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=None),
                                      config=types.SimpleNamespace(eos_token_id=9))
        self.assertEqual(eos_ids(model), [])
        model.generation_config.eos_token_id = [9, 10]
        self.assertEqual(eos_ids(model), [9, 10])

    def test_exact_prefix_is_removed_by_tokens_not_strings(self):
        self.assertEqual(check_generation_boundary([1, 2, 3], [1, 2, 3, 2, 3, 9]), [2, 3, 9])
        self.assertEqual(check_generation_boundary([1, 2, 3], [1, 2, 3]), [])
        for output in ([7, 8], [1, 8, 3, 4]):
            with self.assertRaisesRegex(ValueError, 'exact input prefix'):
                check_generation_boundary([1, 2, 3], output)

    def test_trace_preserves_raw_sequence_and_special_tokens(self):
        tokenizer = CharacterTokenizer()
        prompt = tokenizer.encode('instruction')
        new = tokenizer.encode('instruction') + [9]
        trace = trace_generation(tokenizer, prompt, prompt + new, 'User: instruction\nAssistant:', 'instruction', [9])
        self.assertEqual(trace['new_token_ids'], new)
        self.assertIn('<eot>', trace['new_output_with_special_tokens'])
        self.assertTrue(trace['ended_with_eos'])
        self.assertTrue(trace['input_prefix_matches'])
        self.assertEqual(tokenizer.decode(new, skip_special_tokens=True), 'instruction')
        broken = trace_generation(tokenizer, prompt, [1, 2], 'rendered', 'instruction', [9])
        self.assertIsNone(broken['new_token_ids'])
        self.assertEqual(broken['full_output_token_ids'], [1, 2])

    def test_echo_flag_is_explicit_and_does_not_rewrite_text(self):
        prompt = 'one two three four five six seven eight nine ten'
        text = 'three four five six seven eight nine ten'
        self.assertTrue(echo_observation(text, prompt)['echo_flag'])
        self.assertFalse(echo_observation('', prompt)['echo_flag'])
        self.assertFalse(echo_observation('{"items":[]}', prompt)['echo_flag'])
        self.assertEqual(text, 'three four five six seven eight nine ten')

    def test_real_predict_path_keeps_generated_echo_only(self):
        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data)
            @property
            def shape(self):
                return self.data.shape
            def __getitem__(self, key):
                return Tensor(self.data[key])
            def tolist(self):
                return self.data.tolist()
        class Batch(dict):
            def to(self, device):
                return self
        tokenizer = CharacterTokenizer()
        input_ids = [1, 2, 3]
        new_ids = tokenizer.encode('copied prompt') + [9]
        class Processor:
            def __init__(self):
                self.tokenizer = tokenizer
            def apply_chat_template(self, messages, add_generation_prompt):
                self.messages = messages
                return 'User: copied prompt<eot>\nAssistant:'
            def __call__(self, **kwargs):
                self.inputs = kwargs
                return Batch(input_ids=Tensor([input_ids]), pixel_values=Tensor(np.zeros((1, 1, 3, 2, 2))))
            def batch_decode(self, tensor, **kwargs):
                return [tokenizer.decode(row, **kwargs) for row in tensor.tolist()]
        class Model:
            device = 'fixture_device'
            config = types.SimpleNamespace(eos_token_id=9)
            generation_config = types.SimpleNamespace(eos_token_id=9)
            def eval(self):
                pass
            def generate(self, **kwargs):
                return Tensor([input_ids + new_ids])
        image = types.SimpleNamespace(close=lambda: None)
        processor = Processor()
        predictor = TransformersDietModel(Model(), processor, prompt='copied prompt', capture_trace=True)
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(no_grad=contextlib.nullcontext)}), patch(
                'task2_diet_record.interfaces.load_image', return_value=image):
            self.assertEqual(predictor.predict('fixture.png'), 'copied prompt')
        self.assertEqual(predictor.last_trace['new_token_ids'], new_ids)
        self.assertEqual(processor.messages[0]['role'], 'user')
        self.assertFalse(processor.inputs['truncation'])


class SupervisionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.target = json.dumps(EXAMPLE, separators=(',', ':'))
        self.prompt = 'User: image and request<eot>\nAssistant:'
        self.closure = self.tokenizer.encode('<eot>\n')
        self.ids = [1, 2, 3] + self.tokenizer.encode(' ' + self.target) + self.closure
        self.labels = [-100] * 3 + self.ids[3:]

    def inspect(self, **changes):
        args = {'tokenizer': self.tokenizer, 'ids': self.ids, 'labels': self.labels,
                'attention': [1] * len(self.ids), 'inference_ids': [1, 2, 3], 'rendered_prompt': self.prompt,
                'rendered_full': self.prompt + ' ' + self.target + '<eot>\n', 'target_text': self.target,
                'image_token_id': 2, 'stop_ids': [9], 'max_length': 4096}
        return inspect_supervision(**{**args, **changes})

    def test_complete_answer_and_termination(self):
        report = self.inspect()
        self.assertTrue(report['all_checks_passed'])
        self.assertTrue(report['checks']['termination_included_and_supervised'])

    def test_matching_json_does_not_prove_prompt_masking(self):
        labels = self.labels.copy(); labels[0] = 1
        report = self.inspect(labels=labels)
        self.assertTrue(report['checks']['supervised_json_matches_target'])
        self.assertFalse(report['checks']['only_answer_supervised'])

    def test_image_masking_is_checked_separately(self):
        labels = self.labels.copy(); labels[1] = 2
        self.assertFalse(self.inspect(labels=labels)['checks']['image_tokens_present_and_masked'])

    def test_matching_json_without_eos_is_not_a_pass(self):
        labels = self.labels.copy(); labels[-2] = -100
        report = self.inspect(labels=labels)
        self.assertTrue(report['checks']['supervised_json_matches_target'])
        self.assertFalse(report['checks']['termination_included_and_supervised'])
        self.assertFalse(self.inspect(stop_ids=[8])['checks']['termination_matches_generation_eos'])

    def test_truncation_or_wrong_prefix_is_reported(self):
        self.assertFalse(self.inspect(max_length=5)['checks']['not_truncated'])
        self.assertFalse(self.inspect(inference_ids=[1, 2, 4])['checks']['prompt_is_exact_token_prefix'])

    def test_all_sixteen_rows_checked_even_after_failure(self):
        rows = [{'dish_id': str(i), 'image': f'images/train/{i}.png'} for i in range(16)]
        effects = [{'all_checks_passed': True}] * 3 + [ValueError('fixture failure')] + [{'all_checks_passed': True}] * 12
        with patch('task2_diet_record.supervision_audit.audit_one', side_effect=effects) as audit:
            report = audit_training_rows(None, rows, 'fixture', 2, [9])
        self.assertEqual(audit.call_count, 16)
        self.assertEqual(report['passed_samples'], 15)
        self.assertEqual(len(report['samples_report']), 16)
        self.assertIn('fixture failure', report['samples_report'][3]['error'])


class PromptComparisonTests(unittest.TestCase):
    def test_cli_will_not_write_diagnostics_inside_old_experiment(self):
        Path('outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='outputs') as directory:
            old = Path(directory).resolve()
            args = ['probe', '--model-path', 'fixture_base', '--revision', 'a' * 40,
                    '--data-root', 'fixture_data', '--existing-stage25-dir', str(old), '--output', str(old / 'probe')]
            with patch.object(sys, 'argv', args), patch('task2_diet_record.schema_experiment.completed_stage25',
                    return_value={'experiment_dir': str(old), 'checkpoint': str(old / 'checkpoint') } ):
                with self.assertRaisesRegex(ValueError, 'outside the old'):
                    diagnostic_main()
            self.assertFalse((old / 'probe').exists())

    def test_trained_checkpoint_cannot_be_mixed_into_b_c(self):
        rows = [{'split': 'train'}]
        with self.assertRaisesRegex(ValueError, 'do not mix'):
            compare_prompts(None, None, rows, '.', 'outputs/should_not_be_created', PROMPTS, 512, 'old_adapter')
        with self.assertRaisesRegex(ValueError, 'never test'):
            compare_prompts(None, None, [{'split': 'test'}], '.', 'outputs/should_not_be_created', PROMPTS, 512, 'pretrained_no_adapter')

    def test_three_prompts_are_distinct_without_changing_default(self):
        self.assertEqual(PROMPTS['A'], PROMPT)
        self.assertLess(len(PROMPTS['B']), len(PROMPTS['A']))
        self.assertEqual(len({prompt_hash(p) for p in PROMPTS.values()}), 3)
        self.assertIn('not the answer', PROMPTS['C'])

    def test_comparison_writes_raw_outputs_and_manifests(self):
        class Predictor:
            last_generation = {'generated_tokens': 80}
            last_trace = {'input_prefix_matches': True}
            def predict(self, image):
                return 'unmodified fixture echo text'
        rows = [{'dish_id': s, 'split': s, 'image': f'images/{s}/fixture.png', 'target': EXAMPLE} for s in ('train', 'val')]
        Path('outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='outputs') as directory, patch(
                'task2_diet_record.prompt_diagnostics.TransformersDietModel', return_value=Predictor()):
            output = Path(directory) / 'zero_shot'
            compare_prompts(None, None, rows, directory, output, PROMPTS, 512, 'pretrained_no_adapter')
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['cohort'], 'pretrained_no_adapter')
            for key, prompt in PROMPTS.items():
                report = json.loads((output / f'{key}.json').read_text())
                self.assertEqual(report['predictions'][0]['prediction'], 'unmodified fixture echo text')
                self.assertEqual(report['prompt_sha256'], prompt_hash(prompt))
                self.assertEqual(report['summary']['val']['metrics']['failed_predictions'], 1)
            with self.assertRaises(FileExistsError):
                compare_prompts(None, None, rows, directory, output, PROMPTS, 512, 'pretrained_no_adapter')

    def test_diagnostic_notebook_has_no_training_command(self):
        import ast
        notebook = json.loads(Path('colab/member_c_prompt_diagnostics.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        ast.parse(code)
        self.assertIn('task2_diet_record.prompt_diagnostics', code)
        self.assertIn('prompt_diagnostics_', code)
        self.assertNotIn('run_module("task2_diet_record.train"', code)
        self.assertNotIn('run_module("task2_diet_record.schema_experiment"', code)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
