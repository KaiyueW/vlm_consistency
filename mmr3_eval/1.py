"""Debug one sample: runs the SAME model and the SAME loading code as run_inference.py on a few prompt variants,
so you can see whether the odd image-only answers come from the model or from the files/pipeline.

    python check_one.py --model llava --task logiqa --id 10
    python check_one.py --model llava --task logiqa --id 10 --id 100 --id 1000

Variants printed for each id:
    text            question text only                       (what run_inference / run_ttc send)
    image           image only, no text                      (what run_inference / run_ttc send)
    image+letter    image + "Answer with only the letter of the correct choice."      (extra, for diagnosis)
    image+read      image + "Read out the text in this image."                         (extra: can the model read it?)
"""
import argparse
import os

from PIL import Image

from dataset_utils import load_task
from vlm_models import MODELS, get_model

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--task", required=True)
ap.add_argument("--id", action="append", required=True, help="item id, can be repeated")
ap.add_argument("--data_root", default=os.path.join(HERE, "..", "data"))
ap.add_argument("--max_new_tokens", type=int, default=100)
ap.add_argument("--show", type=int, default=400, help="characters of each answer to print")
args = ap.parse_args()

items = {it["id"]: it for it in load_task(args.data_root, args.task)}
model = get_model(args.model)

for i in args.id:
    if i not in items:
        print(f"id {i} not found in {args.task}")
        continue
    it = items[i]
    image = Image.open(it["image_path"]).convert("RGB")
    print("=" * 80)
    print(f"id {i} | image file: {it['image_path']} | size {image.size} mode {image.mode} | GT: {it['answer']!r}")
    print(f"question (first 200 chars): {it['question'][:200]!r}")

    runs = [
        ("text", None, it["question"]),
        ("image", image, None),
        ("image+letter", image, "Answer with only the letter of the correct choice."),
        ("image+read", image, "Read out the text in this image."),
    ]
    for name, img, txt in runs:
        out = model.generate(image=img, text=txt, max_new_tokens=args.max_new_tokens)
        print(f"--- {name}: {out[:args.show]!r}")