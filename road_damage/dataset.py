"""Convert RDD Pascal VOC XML to YOLO with reproducible, duplicate-safe splits."""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from defusedxml import ElementTree as ET
from PIL import Image

from road_damage import CLASSES

CLASS_IDS = {v: k for k, v in CLASSES.items()}


def parse_annotation(xml, image):
    root = ET.parse(xml).getroot()
    with Image.open(image) as im:
        width, height = im.size
        im.verify()
    size = root.find("size")
    if size is not None:
        if (int(size.findtext("width")), int(size.findtext("height"))) != (width, height):
            raise ValueError(f"Annotation/image size mismatch: {xml}")
    labels = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_IDS:
            raise ValueError(f"Unknown class {name!r} in {xml}; expected D00/D10/D20/D40")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box: {xml}")
        # VOC is 1-based inclusive, but RDD also uses 0 for image-edge minima.
        try:
            xmin, ymin, x2, y2 = (float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing or nonnumeric bounding-box coordinate: {xml}") from exc
        if not all(math.isfinite(v) for v in (xmin, ymin, x2, y2)):
            raise ValueError(f"Non-finite bounding-box coordinate: {xml}")
        if xmin < 0 or ymin < 0:
            raise ValueError(f"Negative bounding-box coordinate: {xml}")
        x1, y1 = max(0, xmin - 1), max(0, ymin - 1)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid or out-of-image bounding box: {xml}")
        labels.append((CLASS_IDS[name], (x1+x2)/(2*width), (y1+y2)/(2*height),
                       (x2-x1)/width, (y2-y1)/height))
    return sorted(labels)


def prepare(source, output, seed=42, ratios=(0.8, 0.1, 0.1), groups_csv=None,
            unsupported_classes="skip-image"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset directory does not exist: {source}")
    if unsupported_classes not in {"skip-image", "error"}:
        raise ValueError("unsupported_classes must be skip-image or error")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be separate, non-overlapping directories")
    if len(ratios) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratios) or abs(sum(ratios)-1) > 1e-6:
        raise ValueError("Three positive split ratios must sum to 1")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a new output directory: {output}")
    images = defaultdict(list)
    for p in sorted(source.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[p.name.casefold()].append(p)
    group_map = {}
    if groups_csv:
        with open(groups_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                group_map[row["image"].replace("\\", "/")] = row["group"]
    records, seen, duplicates = [], {}, 0
    skipped, unsupported_counts = [], Counter()
    for xml in sorted(source.rglob("*.xml")):
        root = ET.parse(xml).getroot()
        unsupported = Counter((obj.findtext("name") or "").strip()
                              for obj in root.findall("object")
                              if (obj.findtext("name") or "").strip() not in CLASS_IDS)
        if unsupported:
            if unsupported_classes == "error":
                raise ValueError(f"Unsupported classes {dict(unsupported)} in {xml}; expected D00/D10/D20/D40")
            unsupported_counts.update(unsupported)
            skipped.append(dict(annotation=xml.relative_to(source).as_posix(),
                                reason="unsupported_classes", classes=dict(unsupported)))
            continue
        filename = Path((root.findtext("filename") or (xml.stem + ".jpg")).replace("\\", "/")).name
        candidates = images.get(filename.casefold(), [])
        adjacent = [p for p in candidates if p.parent == xml.parent]
        if adjacent:
            candidates = adjacent
        # Prefer an image in the same country/subset over ambiguous global filenames.
        for ancestor in xml.parents:
            local = [p for p in candidates if ancestor in p.parents]
            if local:
                candidates = local
                break
            if ancestor == source:
                break
        if len(candidates) != 1:
            raise ValueError(f"Expected one matching image for {xml}, found {len(candidates)}")
        image = candidates[0]
        labels = parse_annotation(xml, image)
        with Image.open(image) as im:
            rgb = im.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        relative = image.relative_to(source).as_posix()
        if groups_csv and relative not in group_map:
            raise ValueError(f"Missing sequence group for {relative}")
        group = group_map[relative] if groups_csv else digest
        if digest in seen:
            prior = seen[digest]
            if prior["labels"] != labels:
                raise ValueError(f"Duplicate image has conflicting labels: {image}")
            if groups_csv and prior["group"] != group:
                raise ValueError(f"Duplicate image assigned to different groups: {image}")
            duplicates += 1
            continue
        record = dict(image=image, relative=relative, labels=labels, digest=digest, group=group)
        seen[digest] = record
        records.append(record)
    if not records:
        raise ValueError(f"No annotated XML/image pairs eligible for the four-class task; "
                         f"skipped {len(skipped)} images, unsupported labels: {dict(unsupported_counts)}")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    keys = sorted(grouped)
    if len(keys) < 3:
        raise ValueError("At least three distinct image/sequence groups are needed")
    random.Random(seed).shuffle(keys)
    # Group counts are allocated deterministically; image ratios may vary with sequence size.
    n = len(keys)
    n_train = max(1, min(n-2, int(n * ratios[0])))
    n_val = max(1, min(n-n_train-1, int(n * ratios[1])))
    split_keys = {"train": keys[:n_train], "val": keys[n_train:n_train+n_val],
                  "test": keys[n_train+n_val:]}
    manifest, counts, class_counts = [], {}, {}
    for split, selected in split_keys.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts[split], counter = 0, Counter()
        for key in selected:
            for record in grouped[key]:
                name = record["digest"][:24]
                shutil.copy2(record["image"], output / "images" / split / (name + record["image"].suffix.lower()))
                lines = [f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}" for c,x,y,w,h in record["labels"]]
                (output / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                counter.update(CLASSES[row[0]] for row in record["labels"])
                counts[split] += 1
                manifest.append(dict(source=record["relative"], digest=record["digest"], group=key, split=split))
        class_counts[split] = {name: counter[name] for name in CLASSES.values()}
    config = dict(path=output.as_posix(), train="images/train", val="images/val", test="images/test", names=CLASSES)
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = dict(seed=seed, requested_ratios=ratios, images=counts, objects=class_counts,
                  duplicates_removed=duplicates, sequence_grouping=bool(groups_csv),
                  unsupported_class_policy=unsupported_classes,
                  skipped_images=len(skipped), unsupported_objects=dict(unsupported_counts),
                  skipped_annotations=skipped, manifest=manifest)
    (output / "split_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"manifest", "skipped_annotations"}}, indent=2))
    print(f"Dataset configuration: {output / 'data.yaml'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/rdd"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ratios", nargs=3, type=float, default=(.8,.1,.1))
    parser.add_argument("--groups-csv", type=Path, help="CSV columns: image (relative path),group (road sequence)")
    parser.add_argument("--unsupported-classes", choices=["skip-image", "error"], default="skip-image",
                        help="Skip and report images with non-target labels (default), or fail strictly")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.seed, args.ratios, args.groups_csv, args.unsupported_classes)
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
