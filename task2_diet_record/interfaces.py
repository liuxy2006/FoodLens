"""所有真实模型只接收图片，不接收 dish_id 或真值。"""
from typing import Protocol, Callable
from pathlib import Path
from .data import load_image

PROMPT = ('Estimate the ingredients and nutrition in this food image. Return only one JSON object. '
          'Use exactly these top-level keys: "items", "total_weight_g", "total_calories_kcal", '
          '"protein_g", "fat_g", "carbohydrate_g". "items" must be an array of objects, each '
          'with exactly "name" (a nonempty string) and "weight_g" (a nonnegative JSON number). '
          'All five top-level nutrition/weight values must be nonnegative JSON numbers, not strings. '
          'Do not rename keys, add fields, or include explanations or Markdown.')


class DietModel(Protocol):
    def predict(self, image: Path) -> str:
        """返回原始生成文本，评价层负责记录失败。"""
        ...


class CaptionModel(Protocol):
    def caption(self, image: Path) -> str: ...


class CaptionDietAdapter:
    """由 A/B 提供 caption 模型，再注入可复现的信息提取函数。"""
    def __init__(self, caption_model: CaptionModel, extractor: Callable[[str], str]):
        self.caption_model = caption_model
        self.extractor = extractor

    def predict(self, image):
        return self.extractor(self.caption_model.caption(Path(image)))


def user_messages(prompt=PROMPT):
    return [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]


def image_prompt_inputs(processor, image, prompt=PROMPT):
    """保持官方的先渲染聊天模板、再传图片给 Processor 的调用方式。"""
    rendered = processor.apply_chat_template(user_messages(prompt), add_generation_prompt=True)
    return rendered, processor(text=[rendered], images=[[image]], truncation=False, return_tensors='pt')


class TransformersDietModel:
    def __init__(self, model, processor, max_new_tokens=1024, prompt=PROMPT, capture_trace=False):
        self.model, self.processor = model, processor
        self.max_new_tokens = max_new_tokens
        self.prompt, self.capture_trace = prompt, capture_trace
        self.last_trace = None

    @classmethod
    def from_pretrained(cls, model_path, adapter=None, device='cpu', revision=None):
        from .vlm import load_backbone
        model, processor = load_backbone(model_path, device, revision, processor_path=adapter)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        return cls(model, processor)

    def predict(self, image):
        import torch
        from .generation_trace import check_generation_boundary, trace_generation, eos_ids
        self.last_trace = None
        self.last_generation = {}
        rgb = load_image(image)
        try:
            rendered, inputs = image_prompt_inputs(self.processor, rgb, self.prompt)
            inputs = inputs.to(self.model.device)
            self.model.eval()
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                                do_sample=False, use_cache=True)
            generated = getattr(generated, 'sequences', generated)
            input_ids, output_ids = inputs['input_ids'][0].tolist(), generated[0].tolist()
            if self.capture_trace:
                self.last_trace = trace_generation(self.processor.tokenizer, input_ids, output_ids,
                                                   rendered, self.prompt, eos_ids(self.model))
                self.last_trace['pixel_values_shape'] = list(inputs['pixel_values'].shape)
                generation_config = getattr(self.model, 'generation_config', None)
                self.last_trace['generation_config'] = {key: getattr(generation_config, key, None)
                    for key in ('eos_token_id', 'bos_token_id', 'pad_token_id', 'num_beams', 'return_dict_in_generate')}
                self.last_trace['other_eos_sources'] = {
                    'model_config': getattr(self.model.config, 'eos_token_id', None),
                    'text_config': getattr(getattr(self.model.config, 'text_config', None), 'eos_token_id', None),
                    'tokenizer': getattr(self.processor.tokenizer, 'eos_token_id', None)}
            # 只在完整 token 前缀严格一致时按长度截取，不用字符串删除提示词。
            check_generation_boundary(input_ids, output_ids)
        finally:
            rgb.close()
        count = generated.shape[1] - inputs['input_ids'].shape[1]
        self.last_generation = {'generated_tokens': count, 'reached_token_limit': count >= self.max_new_tokens,
                                'input_tokens': len(input_ids), 'input_prefix_matches': True}
        return self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
