"""Runs a VLM on every item of a task, once per variant, and saves the raw outputs.

    python run_inference.py --model llava --task logiqa
    python run_inference.py --model qwen  --task all --limit 50

Variants:
    text  : the question text is the prompt, with no image
    image : the question image is the input, with no text prompt (unless --image_prompt is set)

Outputs: outputs/<model>/<task>/<id>_text.txt and <id>_image.txt (0-d string arrays).
Re-running skips files that already exist, so an interrupted run can be resumed.
"""
import argparse
import os
import traceback

import numpy as np
from PIL import Image

from dataset_utils import list_tasks, load_task
from vlm_models import MODELS, get_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, help="task folder name under data/, or 'all'")
    ap.add_argument("--data_root", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--variants", nargs="+", default=["text", "image"], choices=["text", "image"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items per task")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--model_id", default=None, help="override the HF model id")
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28, help="Qwen image budget")
    ap.add_argument("--text_prefix", default="", help="optional text put before the question (default: none)")
    ap.add_argument("--image_prompt", default="",
                    help="optional instruction sent with the image (default: none)")
    args = ap.parse_args()

    tasks = list_tasks(args.data_root) if args.task == "all" else [args.task]
    model = get_model(args.model, args.model_id, max_pixels=args.max_pixels)

    for task in tasks:
        items = load_task(args.data_root, task, args.limit)
        out_dir = os.path.join(args.out_root, args.model, task)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{task}] {len(items)} items -> {out_dir}")

        for n, item in enumerate(items):
            for variant in args.variants:
                path = os.path.join(out_dir, f"{item['id']}_{variant}.txt")
                if os.path.exists(path):
                    continue
                try:
                    if variant == "text":
                        out = model.generate(
                            image=None,
                            text=args.text_prefix + item["question"],
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        image = Image.open(item["image_path"]).convert("RGB")
                        out = model.generate(
                            image=image,
                            text=args.image_prompt or None,
                            max_new_tokens=args.max_new_tokens,
                        )
                    np.save(path, np.array(out))
                except Exception:
                    with open(os.path.join(out_dir, "errors.txt"), "a") as fh:
                        fh.write(f"{item['id']} {variant}\n{traceback.format_exc()}\n")
                    print(f"  error on {item['id']} ({variant}); see errors.txt")
            if (n + 1) % 25 == 0:
                print(f"******{n + 1}/{len(items)}******")


if __name__ == "__main__":
    main()