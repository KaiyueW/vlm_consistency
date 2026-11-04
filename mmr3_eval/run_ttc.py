"""Runs TTC (test-time consistency adaptation) on the 2-variant task (text prompt vs image) and saves the outputs
in the same layout run_inference.py uses, so evaluate.py scores them unchanged.

    python run_ttc.py --model llava --task logiqa --alpha 1 --beta 1 --steps 4 --limit 20     # smoke test
    python run_ttc.py --model qwen  --task all    --alpha 1 --beta 1 --steps 4

Outputs (out_root defaults to ./outputs):
    <model>_ttc_a{alpha}_b{beta}_u{steps}/<task>/<id>_text.txt, <id>_image.txt   adapted answers (the TTC result)
    <model>_ttc_a{alpha}_b{beta}_u{steps}/<task>/<id>_meta.json                  pseudo-label, losses, every step's answers
    <model>_ttc_ori/<task>/<id>_text.txt, <id>_image.txt                         the SAME model before any update

Score with:  python evaluate.py --model <that folder name> --task <task>
Compare TTC against <model>_ttc_ori: same weights/precision/prompts, only the adaptation differs.
Re-running skips samples whose _meta.json already exists.

Lines marked <changed> differ from the original TTC script.
"""
import argparse
import json
import os
import traceback

# <changed> new: send the Hugging Face / pip caches to the lab research disk (the home quota is full).
# This must run before transformers is imported. A value already set in the shell wins.
_RAID_CACHE = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"
os.environ.setdefault("HF_HOME", _RAID_CACHE + "/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", _RAID_CACHE)

import torch
from PIL import Image

from dataset_utils import list_tasks, load_task
from ttc_utils import VARIANTS, adapt_sample, build_inputs
from vlm_models import MODELS, get_model

HERE = os.path.dirname(os.path.abspath(__file__))


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, help="task folder name under data/, or 'all'")
    ap.add_argument("--data_root", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the cross-entropy agreement loss (TTC -a)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the pseudo-label loss (TTC -b)")
    ap.add_argument("--steps", type=int, default=4, help="adaptation steps per sample (TTC -u)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--anchor", default="text", choices=VARIANTS,
                    help="which variant's answer is the pseudo-label (default: text)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="TTC runs in float32; use bfloat16 only if the GPU is too small")
    ap.add_argument("--start", type=int, default=0, help="skip the first N items (to split a task over GPUs)")
    ap.add_argument("--limit", type=int, default=None, help="only N items per task (after --start)")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--model_id", default=None, help="override the HF model id")
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28, help="Qwen image budget")
    ap.add_argument("--text_prefix", default="", help="optional text put before the question (default: none)")
    ap.add_argument("--image_prompt", default="", help="optional instruction sent with the image (default: none)")
    args = ap.parse_args()

    tag = f"{args.model}_ttc_a{args.alpha}_b{args.beta}_u{args.steps}"
    if args.anchor != "text":
        tag += f"_anchor-{args.anchor}"
    ori_tag = f"{args.model}_ttc_ori"

    # <changed> the model is loaded once and reused for every sample (TTC starts a new process per sample)
    vlm = get_model(args.model, args.model_id, max_pixels=args.max_pixels)
    dtype = getattr(torch, args.dtype)
    model = vlm.model.to(dtype).eval()  # TTC also converts to float32
    processor = vlm.processor

    # Only lm_head is trained (same as TTC)
    params = []
    for name, p in model.named_parameters():
        p.requires_grad = "lm_head" in name
        if p.requires_grad:
            params.append(p)
    assert params, "no lm_head parameter found (tied embeddings?)"
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    # <changed> weights (and a fresh optimizer, inside adapt_sample) are reset after every sample, in-process
    snapshot = [p.detach().clone() for p in params]

    tasks = list_tasks(args.data_root) if args.task == "all" else [args.task]
    for task in tasks:
        items = load_task(args.data_root, task)[args.start:]
        if args.limit:
            items = items[:args.limit]
        out_dir = os.path.join(args.out_root, tag, task)
        ori_dir = os.path.join(args.out_root, ori_tag, task)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ori_dir, exist_ok=True)
        print(f"[{task}] {len(items)} items -> {out_dir}")

        for n, item in enumerate(items):
            meta_path = os.path.join(out_dir, f"{item['id']}_meta.json")
            if os.path.exists(meta_path):
                continue
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                raw = {"text": (None, args.text_prefix + item["question"]),
                       "image": (image, args.image_prompt or None)}
                inputs = {}
                for v in VARIANTS:
                    prompt, inputs[v] = build_inputs(args.model, processor, raw[v][0], raw[v][1], model.device, dtype)
                    vlm._show(prompt, raw[v][0])  # prints the first few prompts

                res = adapt_sample(
                    model, processor.tokenizer, inputs, params, snapshot,
                    anchor=args.anchor, alpha=args.alpha, beta=args.beta, steps=args.steps,
                    lr=args.lr, weight_decay=args.weight_decay, max_new_tokens=args.max_new_tokens,
                )

                for v in VARIANTS:
                    write_text(os.path.join(ori_dir, f"{item['id']}_{v}.txt"), res["ori"][v])
                    write_text(os.path.join(out_dir, f"{item['id']}_{v}.txt"), res["final"][v])
                res["id"] = item["id"]
                write_text(meta_path, json.dumps(res, indent=2, ensure_ascii=False))  # written last = "done"

                first, last = (res["losses"][0], res["losses"][-1]) if res["losses"] else ({}, {})
                print(f"  {item['id']}: pseudo={res['pseudo_label']!r} best_step={res['best_step']} "
                      f"loss {first.get('total', float('nan')):.3f} -> {last.get('total', float('nan')):.3f} {res['note']}")
            except Exception:
                with open(os.path.join(out_dir, "errors.txt"), "a") as fh:
                    fh.write(f"{item['id']}\n{traceback.format_exc()}\n")
                print(f"  error on {item['id']}; see errors.txt")
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(items)}")


if __name__ == "__main__":
    main()