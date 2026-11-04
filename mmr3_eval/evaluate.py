"""Scores saved outputs with the metrics used in MM-R3 / TTC, adapted to 2 variants (text, image).

    python evaluate.py --model llava --task logiqa
    python evaluate.py --model qwen  --task all

Per item there are two outputs (text-input, image-input). Metrics:
    accuracy        GT appears in the output (plain substring, as in TTC)
    soft accuracy   fuzz.token_set_ratio(GT, output) >= 85          (TTC script)
    ss_gt           cosine similarity (all-MiniLM-L6-v2) between output and GT
    consistency     cosine similarity between the text-output and image-output > threshold (0.7)
    ss              the same cosine similarity, unthresholded
    exact_agree     the two outputs are identical (stricter than the 0.7 threshold)
    hmean           TTC's Hmean = harmonic mean of [(soft + ss_gt)/2, (consistency + ss)/2], 0-100
Also reported: the failure subset (items where not both outputs are correct), which shows
whether failures are stable or flipping between text and image.
"""
import argparse
import json
import os
import statistics


import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

from dataset_utils import list_tasks, load_task

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["text", "image"]


def clean(s):
   # clean the outputs
    return str(s).replace("</s>", "").lower()


def is_correct(gt, out):
    return gt in out  # plain substring match


def load_outputs(out_dir, items):
    kept, outs = [], []
    for it in items:
        paths = [os.path.join(out_dir, f"{it['id']}_{v}.txt") for v in VARIANTS]
        if all(os.path.exists(p) for p in paths):
            kept.append(it)
            outs.append([open(p, encoding="utf-8").read() for p in paths])
    return kept, outs


def evaluate_task(args, model, task, sbert):
    items = load_task(args.data_root, task)
    out_dir = os.path.join(args.out_root, model, task)
    items, raw = load_outputs(out_dir, items)
    n = len(items)
    if n == 0:
        print(f"[{model}/{task}] no complete outputs found in {out_dir}")
        return None

    outs = [[clean(o) for o in pair] for pair in raw]  # (n, 2)
    gts = [clean(it["answer"]) for it in items]

    correct = np.array([[is_correct(g, o) for o in pair] for g, pair in zip(gts, outs)], float) #(n, 2)
    soft = np.array([[fuzz.token_set_ratio(g, o) >= 85 for o in pair] for g, pair in zip(gts, outs)], float) #(n, 2)

    flat = [o for pair in outs for o in pair] #(2n)
    e_out = sbert.encode(flat, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    e_out = e_out.view(n, 2, -1)
    e_gt = sbert.encode(gts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    ss_gt = (e_out * e_gt.unsqueeze(1)).sum(-1).cpu().numpy()  # (n, 2) cosine similarity btw outputs and gts
    sim = (e_out[:, 0] * e_out[:, 1]).sum(-1).cpu().numpy()  # (n,) cosine similarity btw outputs

    consistent = (sim > args.threshold).astype(float) #(n, )
    exact = np.array([pair[0].strip() == pair[1].strip() for pair in outs], float) # (n, )

    per_item = {
        "acc_text": correct[:, 0], "acc_image": correct[:, 1], "acc_mean": correct.mean(1),
        "soft_text": soft[:, 0], "soft_image": soft[:, 1], "soft_mean": soft.mean(1),
        "ss_gt_text": ss_gt[:, 0], "ss_gt_image": ss_gt[:, 1], "ss_gt_mean": ss_gt.mean(1),
        "consistency_acc": consistent, "consistency_sim": sim, "exact_agree": exact,
    }
    metrics = {k: float(v.mean()) for k, v in per_item.items()}

    a = (metrics["acc_mean"] + metrics["ss_gt_mean"]) / 2 * 100
    b = (metrics["consistency_acc"] + metrics["consistency_sim"]) / 2 * 100
    hmean = statistics.harmonic_mean([a, b]) if a > 0 and b > 0 else 0.0

    summary = {"model": model, "task": task, "n_items": n, "threshold": args.threshold,
               "metrics": metrics, "hmean": float(hmean)}

    fail = ~correct.all(1)
    summary["failure_subset"] = {"n_items": int(fail.sum())}
    if fail.any():
        summary["failure_subset"].update({
            "consistency_acc": float(consistent[fail].mean()),
            "consistency_sim": float(sim[fail].mean()),
            "ss_gt_mean": float(ss_gt[fail].mean()),
        })

    np.save(os.path.join(out_dir, "metrics.npy"), {
        "correct": correct, "soft_correct": soft, "consistent_gt": ss_gt,
        "consistency_sim": sim, "ids": [it["id"] for it in items],
    })

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print_summary(summary)
    return summary


def print_summary(s):
    print(f"\n=== {s['model']} / {s['task']}  (n={s['n_items']}) ===")
    for k, v in s["metrics"].items():
        print(f"{k:14s} {v * 100:6.2f}")
    print(f"{'hmean':14s} {s['hmean']:6.2f}")
    print("failure subset:", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in s["failure_subset"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="task folder name or 'all'")
    ap.add_argument("--data_root", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tasks = list_tasks(args.data_root) if args.task == "all" else [args.task]
    for task in tasks:
        evaluate_task(args, args.model, task, sbert)


if __name__ == "__main__":
    main()