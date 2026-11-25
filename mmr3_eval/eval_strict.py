"""Answer-aware scoring for TTC outputs (ori vs TTC), no torch / sentence-transformers needed (starts in seconds).

    python eval_strict.py --task mmlu logiqa Math_2-51MR_52-101MES-hard
    python eval_strict.py --task all --ttc qwen_ttc_a1.0_b1.0_u4_L40 --ori qwen_ttc_ori

The MM-R3 / TTC metrics (evaluate.py) compare whole strings, which does not fit these tasks:
  * multiple choice: GT "d" is "contained" in almost any long text, and "D. Toilet" vs "D" never matches exactly
  * math: GT "40" is "contained" in "...140..." or in any number list
So here the GT type decides how an output is read:
  GT = one letter   -> the output must START with that letter  ("D", "D. Toilet", "(d)")
  GT = a number     -> \\boxed{} if present, else the LAST number in the output must equal it to the GT's own number of
                       decimals, rounded or truncated (commas, $, fractions ok; a fraction GT allows a 0.1% decimal)
  GT = a word/other -> all the GT words (stemmed) appear in the output, not preceded by "not"
Agreement (no GT needed): same(a, b) = same letter, or numbers within 0.1%, or near-identical text.
"""
import argparse, difflib, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


def _find(folder, i):
    for n in (i, i + ".txt"):
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p


def clean(s):
    s = str(s).replace("</s>", "").lower()
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)  # \frac{16}{3} -> 16/3
    return s.replace("$", " ")


def to_val(tok):
    tok = tok.replace(",", "")
    try:
        if "/" in tok:
            a, b = tok.split("/")
            return float(a) / float(b)
        return float(tok)
    except (ValueError, ZeroDivisionError):
        return None


# <changed> new: a generation that falls into a loop ends in "... wall  4  4  4  4" or "... 5  5  5  5". Cut such a tail
# (one token repeated 3+ times at the very end) before taking "the last number", otherwise the loop's number is the answer.
TAIL = re.compile(r"(?:\s+(\S+))(?:\s+\1){2,}\s*$")


def strip_loop_tail(o):
    return TAIL.sub("", o)


def last_number(o):
    o = strip_loop_tail(o)
    m = re.findall(r"\\boxed\{([^{}]*)\}", o)
    nums = NUM.findall(m[-1]) if m else NUM.findall(o)
    return to_val(nums[-1]) if nums else None


def lead_letter(o):
    m = re.match(r"^\W*([a-z])(?:[.):,]|\s*$)", o.strip())
    if m:
        return m.group(1)
    # <changed> also read "The correct answer is B. ...", "... best represented by option A: ...", "The assumption is: C. ..."
    # (the letter must be followed by . ) : , or the end, so "the answer is a total of 90" is NOT read as the letter a)
    m = re.findall(r"(?:answer|option|choice)\s*(?:is)?\s*[:\s]\s*\(?([a-d])(?:[.):,]|\s*$)", o)
    if not m:
        m = re.findall(r"\bis:\s*\(?([a-d])[.):]", o)
    return m[-1] if m else None


_SUF = ("ically", "ences", "ence", "ents", "ent", "ly", "ing", "ed", "es", "al", "s", "e")


def stem(w):
    """Crude stemmer: converges / converged / convergent -> converg ; absolutely -> absolut ; conditionally -> condition."""
    w = w.lower()
    for _ in range(3):
        for suf in _SUF:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        else:
            break
    return w


def is_correct(gt, out):
    g, o = gt.strip().lower(), clean(out)
    if re.fullmatch(r"[a-z]", g):
        return lead_letter(o) == g
    if re.fullmatch(NUM.pattern, g.replace("$", "")):
        gs = g.replace("$", "").replace(",", "")
        gv, ov = to_val(gs), last_number(o)
        if gv is None or ov is None:
            return False
        if "/" in gs:  # fraction GT: the model may answer with a decimal
            return math.isclose(gv, ov, rel_tol=1e-3, abs_tol=1e-9)
        # decimal / integer GT: the answer must match to the GT's own number of decimals, rounded OR truncated
        # (e.g. GT 15.052 accepts 15.052, 15.0525555..., but not 15.06; GT 106000 needs 106000 exactly)
        d = len(gs.split(".")[1]) if "." in gs else 0
        f = 10 ** d
        return abs(round(ov * f) / f - gv) < 0.5 / f or abs(math.trunc(ov * f + 1e-9) / f - gv) < 0.5 / f
    # <changed> GT mixing digits and text ("40 miles", "2x+3"): the digits matter, so require the whole GT (spaces ignored)
    if re.search(r"\d", g):
        sq = lambda x: "".join(clean(x).replace(",", "").split())
        return sq(g) in sq(o)
    # <changed> word answers ("diverge", "converge absolutely"): compare word STEMS, so "Converges absolutely." and
    # "Conditionally convergent" match "converge absolutely" / "converge conditionally", but a partial answer
    # ("Converges" for "converge absolutely") does not. A "not" right before the answer word makes it wrong.
    gt_stems = [stem(w) for w in re.findall(r"[a-z]+", clean(g))]
    words = re.findall(r"[a-z']+", o)
    out_stems = [stem(w) for w in words]
    if not gt_stems or not set(gt_stems) <= set(out_stems):
        return False
    first = out_stems.index(gt_stems[0])
    return not (first > 0 and words[first - 1] in ("not", "n't", "never", "non"))


def key(out):
    o = clean(out)
    l = lead_letter(o)
    if l:
        return ("L", l)
    n = last_number(o)
    return ("N", round(n, 6)) if n is not None else ("T", " ".join(o.split())[:80])


def same(a, b):
    """<changed> new: do two outputs give the SAME answer? (same letter / numbers within 0.1% / near-identical text)
    Shared by eval_strict.py, evaluate.py (agree_ans) and reselect.py, so agreement means the same thing everywhere."""
    ka, kb = key(a), key(b)
    if ka[0] != kb[0]:
        return False
    if ka[0] == "N":
        return math.isclose(ka[1], kb[1], rel_tol=1e-3, abs_tol=1e-9)
    if ka[0] == "L":
        return ka[1] == kb[1]
    return difflib.SequenceMatcher(None, ka[1], kb[1]).ratio() >= 0.9


def score_folder(folder, items):
    res = {}
    for it in items:
        p = [os.path.join(folder, f"{it['id']}_{v}.txt") for v in ("text", "image")]
        if all(os.path.exists(x) for x in p):
            res[it["id"]] = [_read(x) for x in p]
    return res


def load_items(data_root, task):
    td = os.path.join(data_root, task)
    out = []
    for f in sorted(os.listdir(os.path.join(td, "image"))):
        i, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        a = _find(os.path.join(td, "answer"), i)
        if a:
            out.append({"id": i, "gt": _read(a)})
    return out


def stats(items, outs):
    t = im = ag = both = n = 0
    for it in items:
        o = outs[it["id"]]
        ct, ci = is_correct(it["gt"], o[0]), is_correct(it["gt"], o[1])
        t += ct; im += ci; both += ct and ci; ag += same(o[0], o[1]); n += 1
    return n, t, im, ag, both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", required=True, help="task folder names, or 'all'")
    ap.add_argument("--ttc", default="qwen_ttc_a1.0_b1.0_u4_L40")
    ap.add_argument("--ori", default="qwen_ttc_ori")
    ap.add_argument("--data_root", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    a = ap.parse_args()
    tasks = sorted(os.listdir(a.data_root)) if a.task == ["all"] else a.task

    print(f"{'task':44s} {'n':>4s} | {'text acc':>9s} | {'image acc':>9s} | {'agree':>9s} | {'both ok':>9s}   (ori>ttc, counts)")
    for t in tasks:
        d_ttc, d_ori = os.path.join(a.out_root, a.ttc, t), os.path.join(a.out_root, a.ori, t)
        if not (os.path.isdir(d_ttc) and os.path.isdir(d_ori)):
            print(f"{t:44s}  (no outputs: {'ttc' if not os.path.isdir(d_ttc) else 'ori'} folder missing)")
            continue
        items = load_items(a.data_root, t)
        o_ttc, o_ori = score_folder(d_ttc, items), score_folder(d_ori, items)
        items = [it for it in items if it["id"] in o_ttc and it["id"] in o_ori]
        if not items:
            print(f"{t:44s}  (no complete items)")
            continue
        n, *s_ori = stats(items, o_ori)
        _, *s_ttc = stats(items, o_ttc)
        cells = " | ".join(f"{x:>4d}>{y:<4d}" for x, y in zip(s_ori, s_ttc))
        print(f"{t:44s} {n:>4d} | {cells}")


if __name__ == "__main__":
    main()
