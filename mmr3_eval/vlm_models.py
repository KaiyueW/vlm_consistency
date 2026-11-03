"""Thin wrappers that give LLaVA and Qwen2-VL the same interface:

    model.generate(image=PIL.Image or None, text=str or None, max_new_tokens=100) -> str

Greedy decoding is used everywhere, and only the newly generated tokens are returned.
Both models load once and are reused for every item.
"""
import torch


class BaseVLM:
    n_shown = 0
    max_shown = 4
    def _show(self, prompt, image):
        if BaseVLM.n_shown < BaseVLM.max_shown:
            BaseVLM.n_shown += 1
            print(f"----- prompt (image={'yes' if image is not None else 'no'}) -----\n{prompt}\n-----")

    def generate(self, image=None, text=None, max_new_tokens=100):
        raise NotImplementedError


class LlavaVLM(BaseVLM):
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", **_):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda:0").eval()

    @torch.inference_mode()
    def generate(self, image=None, text=None, max_new_tokens=100):
        content = []
        if image is not None:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        self._show(prompt, image)

        if image is not None:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device, torch.float16)

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


class QwenVLM(BaseVLM):
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", max_pixels=1024 * 28 * 28, **_):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    @torch.inference_mode()
    def generate(self, image=None, text=None, max_new_tokens=100):
        content = []
        if image is not None:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        self._show(prompt, image)

        inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


MODELS = {"llava": LlavaVLM, "qwen": QwenVLM}


def get_model(name, model_id=None, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(MODELS)}")
    if model_id:
        kwargs["model_id"] = model_id
    return MODELS[name](**kwargs)