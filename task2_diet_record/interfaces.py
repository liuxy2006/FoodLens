"""所有真实模型只接收图片，不接收 dish_id 或真值。"""
from typing import Protocol, Callable
from pathlib import Path
from .data import load_image

PROMPT = ('Estimate the ingredients and nutrition in this food image. Return only a JSON object '
          'with items (each containing name and weight_g), total_weight_g, total_calories_kcal, '
          'protein_g, fat_g, carbohydrate_g. Use nonnegative numbers. Do not add other fields.')


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


def user_messages():
    return [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]


class TransformersDietModel:
    def __init__(self, model, processor, max_new_tokens=1024):
        self.model, self.processor = model, processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(cls, model_path, adapter=None, device='cpu', revision=None):
        from .vlm import load_backbone
        model, processor = load_backbone(model_path, device, revision)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        return cls(model, processor)

    def predict(self, image):
        import torch
        rgb = load_image(image)
        prompt = self.processor.apply_chat_template(user_messages(), add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[rgb]], return_tensors='pt')
        inputs = inputs.to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        rgb.close()
        return self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
