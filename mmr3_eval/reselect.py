"""Re-choose the best TTC step from the saved _meta.json files (no model, no GPU, takes seconds).

    python reselect.py --rule ans                 # answer-aware agreement, ties broken by fuzz.ratio
    python reselect.py --rule ans_early           # answer-aware agreement, ties -> EARLIEST step (recommended)
    python reselect.py --rule tie                 # TTC's token_set_ratio, ties broken by the stricter fuzz.ratio
    python reselect.py --src qwen_ttc_a1.0_b1.0_u4_L40 --rule ans

Writes a NEW folder <src>_resel_<rule>/<task>/ with the same files (<id>_text.txt, <id>_image.txt, <id>_meta.json),
so the original outputs are never touched. Score it with:  python eval_strict.py --task ... --ttc <src>_resel_<rule>

Rules (candidates of a step = text answer, image answer, pseudo-label):
  tie  mean pairwise fuzz.token_set_ratio (TTC's rule); ties -> mean pairwise fuzz.ratio; ties -> earlier step
  ans  <changed> share of the 3 pairs that give the SAME extracted answer (same letter / same number / near-identical
       text), so "A" == "A. option text" and 15.052 == 15.0525555...; ties -> mean pairwise fuzz.ratio; ties -> earlier step
  ans_early  same agreement share, but ties -> the EARLIEST step (no fuzz.ratio tie-break): among the steps where the
       answers agree equally well, keep the least adapted one. fuzz.ratio rewards byte-identical strings, which drags
       samples that already agree at step 0 ("A. text" vs "A") to later, more adapted steps for no benefit.
"""
import argparse, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzywuzzy import fuzz
import eval_strict as E

V = ["text", "image"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def same(a, b):
    """Do two outputs give the same answer?"""
    ka, kb = E.key(a), E.key(b)
    if ka[0] != kb[0]:
        return False
    if ka[0] == "N":
        return math.isclose(ka[1], kb[1], rel_tol=1e-3, abs_tol=1e-9)
    if ka[0] == "L":
        return ka[1] == kb[1]
    return fuzz.ratio(ka[1], kb[1]) >= 90


def score(c, rule):
    strict = sum(fuzz.ratio(c[x], c[y]) / 100 for x, y in PAIRS) / 3
    if rule == "ans_early":
        return (sum(same(c[x], c[y]) for x, y in PAIRS) / 3, 0.0)
    if rule == "tie":
        return (sum(fuzz.token_set_ratio(c[x], c[y]) / 100 for x, y in PAIRS) / 3, strict)
    return (sum(same(c[x], c[y]) for x, y in PAIRS) / 3, strict)


def select(steps, pseudo, rule):
    best_key, best_i = (-1.0, -1.0), 0
    for i, a in enumerate(steps):
        k = score([a[v] for v in V] + [pseudo], rule)
        if k > best_key:  # strict >: ties keep the earlier step
            best_key, best_i = k, i
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="qwen_ttc_a1.0_b1.0_u4_L40")
    ap.add_argument("--rule", choices=["tie", "ans", "ans_early"], default="ans")
    ap.add_argument("--out_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    a = ap.parse_args()
    src, dst = os.path.join(a.out_root, a.src), os.path.join(a.out_root, f"{a.src}_resel_{a.rule}")
    print(f"{'task':44s} {'n':>4s} {'changed':>8s} {'old best_step':>28s}   {'new best_step'}")
    for task in sorted(os.listdir(src)):
        os.makedirs(os.path.join(dst, task), exist_ok=True)
        n = ch = skip = 0
        old, new = {}, {}
        for f in glob.glob(os.path.join(src, task, "*_meta.json")):
            m = json.load(open(f)); i = os.path.basename(f).split("_")[0]
            if not m.get("steps") or m["steps"][0] != m["ori"]:  # old format: steps[0] is not the unadapted answer
                skip += 1; continue
            b = select(m["steps"], m["pseudo_label"], a.rule)
            n += 1; ch += b != m["best_step"]
            old[m["best_step"]] = old.get(m["best_step"], 0) + 1; new[b] = new.get(b, 0) + 1
            m["best_step_old"], m["best_step"], m["final"] = m["best_step"], b, m["steps"][b]
            json.dump(m, open(os.path.join(dst, task, f"{i}_meta.json"), "w"), indent=2)
            for v in V:
                open(os.path.join(dst, task, f"{i}_{v}.txt"), "w", encoding="utf-8").write(m["final"][v])
        print(f"{task:44s} {n:>4d} {ch:>8d} {str(dict(sorted(old.items()))):>28s}   {dict(sorted(new.items()))}" + (f"  (skipped {skip} old-format)" if skip else ""))


if __name__ == "__main__":
    main()
