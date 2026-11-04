"""Building blocks for TTC (Test-Time Consistency) adaptation, reduced to 2 variants:

    "text"  : the question is the text prompt, no image
    "image" : the question rendered as an image, no text prompt (unless --image_prompt is set)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzywuzzy import fuzz

VARIANTS = ["text", "image"]
PAD_LEN = 10  # TTC keeps at most 10 answer tokens as the supervision target


class HLoss(nn.Module): # agreement loss
    """Symmetric cross-entropy between two output distributions. Copied from TTC, unchanged."""

    def forward(self, x1, x2):
        p1 = F.softmax(x1, dim=-1)
        p2 = F.softmax(x2, dim=-1)
        ce_1_2 = F.cross_entropy(x1, p2)
        ce_2_1 = F.cross_entropy(x2, p1)
        return (ce_1_2 + ce_2_1) / 2.0


def build_inputs(name, processor, image, text, device, dtype):
    """Same prompt construction as vlm_models.generate(), but returns the processor inputs.

    Returns (prompt_string, inputs). `inputs` is a processor output (input_ids, attention_mask,
    pixel_values / image_grid_thw when an image is given), moved to `device`, floats cast to `dtype`.
    """
    content = []
    if image is not None:
        content.append({"type": "image"})
    if text:
        content.append({"type": "text", "text": text})
    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    if name == "llava":
        if image is not None:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")
    else:  # qwen
        inputs = processor(
            text=[prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        )
    return prompt, inputs.to(device, dtype)


@torch.inference_mode()
def generate_answer(model, tokenizer, inputs, max_new_tokens=100):
    """Greedy answer, only the newly generated tokens."""
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:] # only decode the output answers.
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

## till here!

def label_token_ids(tokenizer, text, device):
    """Pseudo-label string -> token ids (1, n), truncated to PAD_LEN."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :PAD_LEN]
    return ids.to(device)


def forward_on_label(model, inputs, label_ids):
    """One teacher-forced forward pass: prompt + pseudo-label tokens.

    Returns (ce, mean_logits):
      ce           cross-entropy of the pseudo-label tokens given the prompt (the Pseudo-Label Consistency term)
      mean_logits  (1, V) logits averaged over the answer positions (the input of the Agreement term, as in TTC)
    """
    # <changed> TTC appends 10 pad tokens after the prompt and puts the label tokens at those positions, and for
    # llava passes `images=` (not a HF argument name) instead of pixel_values. Here the label tokens are appended
    # right after the prompt (proper teacher forcing) and every processor input (pixel_values, image_grid_thw, ...)
    # is passed through, so the image really enters the loss. The loss is computed here, not by HF `labels=`.
    prompt_ids = inputs["input_ids"]
    P, n = prompt_ids.shape[1], label_ids.shape[1]
    fwd = dict(inputs)
    fwd["input_ids"] = torch.cat([prompt_ids, label_ids], dim=1)
    fwd["attention_mask"] = torch.ones_like(fwd["input_ids"])
    out = model(**fwd, use_cache=False, return_dict=True)
    pred = out.logits[0, P - 1 : P - 1 + n]  # the logits that predict the n label tokens
    ce = F.cross_entropy(pred, label_ids[0])
    return ce, pred.mean(0, keepdim=True)


def select_best_step(step_answers, pseudo_label):
    """Index of the step whose answers (plus the pseudo-label) agree most with each other.

    Same rule as TTC's find_best_answers (mean pairwise fuzz.token_set_ratio, first best wins).
    Uses no ground truth.
    """
    best_avg, best_i = -1.0, 0
    for i, answers in enumerate(step_answers):
        cand = [answers[v] for v in VARIANTS] + [pseudo_label]
        sims = [
            fuzz.token_set_ratio(cand[a], cand[b]) / 100.0
            for a in range(len(cand))
            for b in range(a + 1, len(cand))
        ]
        avg = sum(sims) / len(sims)
        if avg > best_avg:
            best_avg, best_i = avg, i
    return best_i


def adapt_sample(model, tokenizer, inputs, params, snapshot, *, anchor="text", alpha=1.0, beta=1.0,
                 steps=4, lr=5e-4, weight_decay=0.01, max_new_tokens=100):
    """TTC for ONE sample. `inputs` = {"text": ..., "image": ...} (outputs of build_inputs).

    1. greedy answers with the original model          -> result["ori"]
    2. pseudo-label = the `anchor` variant's answer    (<changed>: TTC takes the most frequent answer of 3 variants)
    3. `steps` x [loss = alpha*agreement + beta*pseudo-label; AdamW step on lm_head; regenerate]
    4. pick the most self-consistent step               -> result["final"]
    The trainable `params` are restored from `snapshot` before returning, so samples never influence each other
    (TTC gets this by starting a fresh process per sample).
    """
    ori = {v: generate_answer(model, tokenizer, inputs[v], max_new_tokens) for v in VARIANTS}
    pseudo = ori[anchor]
    result = {"ori": ori, "pseudo_label": pseudo, "anchor": anchor, "alpha": alpha, "beta": beta,
              "steps": [], "losses": [], "best_step": -1, "final": ori, "note": ""}

    label_ids = label_token_ids(tokenizer, pseudo, inputs["text"]["input_ids"].device)
    if label_ids.shape[1] == 0:
        result["note"] = "empty pseudo-label: adaptation skipped, final = original answers"
        return result

    entropy_loss = HLoss()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    try:
        for step in range(steps):
            ce_text, logits_text = forward_on_label(model, inputs["text"], label_ids)
            ce_image, logits_image = forward_on_label(model, inputs["image"], label_ids)

            # K = 2 variants -> the average over all pairs is the single (text, image) pair
            loss_agree = entropy_loss(logits_text, logits_image)
            # the pseudo-label is the supervision target of ALL variants (text included), as in TTC
            loss_pl = (ce_text + ce_image) / 2.0
            loss = alpha * loss_agree + beta * loss_pl

            if not torch.isfinite(loss):
                result["note"] = f"non-finite loss at step {step}: stopped early"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            result["losses"].append({"agree": loss_agree.item(), "pseudo": loss_pl.item(), "total": loss.item()})
            result["steps"].append({v: generate_answer(model, tokenizer, inputs[v], max_new_tokens) for v in VARIANTS})
    finally:
        with torch.no_grad():
            for p, saved in zip(params, snapshot):
                p.copy_(saved)
                p.grad = None

    if result["steps"]:
        result["best_step"] = select_best_step(result["steps"], pseudo)
        result["final"] = result["steps"][result["best_step"]]
    return result