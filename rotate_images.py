import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Rotate all JPG images clockwise by 90 degrees."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input image folder"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output image folder"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in input_dir.glob("*.jpg"):
        with Image.open(image_path) as image:
            rotated = image.transpose(Image.Transpose.ROTATE_270)
            output_path = output_dir / image_path.name
            rotated.save(output_path)

    print(f"Done! Rotated images saved to {output_dir}")


if __name__ == "__main__":
    main()