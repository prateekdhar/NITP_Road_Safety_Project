"""Synthetic plumbing check ONLY. These images do not measure road detection quality."""
import argparse
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from road_damage.dataset import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "pd"], default="pd")
    args = parser.parse_args()
    root = Path("tmp/smoke")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        image = Image.new("RGB", (64,64), (40+i*3,40,40))
        ImageDraw.Draw(image).rectangle((16,16,40,40), outline="white", width=2)
        image.save(raw / f"sample{i}.png")
        (raw / f"sample{i}.xml").write_text(
            f'<annotation><filename>sample{i}.png</filename><size><width>64</width><height>64</height></size>'
            f'<object><name>{["D00","D10","D20","D40"][i%4]}</name><bndbox>'
            '<xmin>17</xmin><ymin>17</ymin><xmax>41</xmax><ymax>41</ymax></bndbox></object></annotation>')
    prepared = root / "prepared"
    if not (prepared / "data.yaml").exists():
        prepare(raw, prepared, ratios=(.5,.25,.25))
    subprocess.run([sys.executable, "-m", "road_damage.train", "--data", str(prepared / "data.yaml"),
                    "--variant", args.variant, "--epochs", "1", "--imgsz", "64", "--batch", "2",
                    "--scratch", "--name", "smoke_" + args.variant], check=True)


if __name__ == "__main__":
    main()
