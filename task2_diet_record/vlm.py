"""SmolVLM 多模态加载、仅答案监督的批处理。依赖按需导入。"""
from .interfaces import user_messages


def load_backbone(model_path, device='cpu', revision=None, allow_download=False):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if device == 'xpu' and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise RuntimeError('Intel XPU unavailable; use CPU or a supported training host')
    kwargs = {'local_files_only': not allow_download, 'revision': revision}
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    processor.tokenizer.padding_side = 'right'
    model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float32,
                                                 attn_implementation='eager', **kwargs)
    # 默认 float32，CPU/Intel 环境不假设支持 CUDA 量化或半精度。
    model.to(device)
    return model, processor


def answer_start(full_ids, prompt_ids):
    """图像 token 展开后仍须严格前缀相同，否则拒绝产生错误监督。"""
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('chat template prompt is not an exact token prefix')
    if len(full_ids) <= len(prompt_ids):
        raise ValueError('no supervised answer tokens')
    return len(prompt_ids)


class SupervisedCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def __call__(self, examples):
        prompts, texts = [], []
        for row in examples:
            messages = user_messages()
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))
            messages = messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': row['target_text']}]}]
            texts.append(self.processor.apply_chat_template(messages, add_generation_prompt=False))
        images = [[row['image']] for row in examples]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors='pt')
        prefix = self.processor(text=prompts, images=images, padding=True, return_tensors='pt')
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError('sequence too long; increase max_length instead of truncating image/answer tokens')
        labels = batch['input_ids'].clone()
        for i in range(len(examples)):
            full_ids = batch['input_ids'][i][batch['attention_mask'][i].bool()].tolist()
            prompt_ids = prefix['input_ids'][i][prefix['attention_mask'][i].bool()].tolist()
            start = answer_start(full_ids, prompt_ids)
            labels[i, :start] = -100
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        return batch
