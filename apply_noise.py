import argparse
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from imagecorruptions import corrupt, get_corruption_names


def parse_args():
    names = get_corruption_names("all")
    p = argparse.ArgumentParser(description="Add a corruption (ImageNet-C style) to all images in a folder.")
    p.add_argument("--input_dir", type=Path, required=True, help="Folder with the original images")
    p.add_argument("--output_dir", type=Path, required=True, help="Folder for corrupted images (created if missing)")
    p.add_argument("--corruption", default="impulse_noise", choices=names, help="Corruption type")
    p.add_argument("--severity", type=int, default=2, choices=range(1, 6), help="Severity level 1-5")
    p.add_argument("--quality", type=int, default=95, help="JPEG save quality")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {".jpg"})
    for path in tqdm(paths, desc=f"{args.corruption} (severity {args.severity})"):
        try:
            img = np.array(Image.open(path).convert("RGB"))

            seed = zlib.crc32(path.stem.encode())  # same image -> same noise
            np.random.seed(seed)
            random.seed(seed)

            out = corrupt(img, corruption_name=args.corruption, severity=args.severity)
            out = np.uint8(np.clip(out, 0, 255))
            Image.fromarray(out).save(args.output_dir / f"{path.stem}.jpg", quality=args.quality)
        except Exception as e:
            tqdm.write(f"[WARN] skipped {path.name}: {e}")

    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()