import json
from pathlib import Path

import pytest
from PIL import Image
from road_damage.dataset import prepare, parse_annotation


def example(root, index=0, box=(1, 1, 40, 20), label="D00"):
    root.mkdir(parents=True, exist_ok=True)
    image = root / f"road{index}.jpg"
    Image.new("RGB", (40, 20), (index*20, index*10, index*5)).save(image)
    xml = root / f"road{index}.xml"
    x1,y1,x2,y2 = box
    xml.write_text(f'<annotation><filename>{image.name}</filename><size><width>40</width><height>20</height></size>'
                   f'<object><name>{label}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>'
                   f'<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object></annotation>')
    return xml, image


def test_voc_conversion(tmp_path):
    xml, image = example(tmp_path)
    assert parse_annotation(xml, image) == [(0, .5, .5, 1., 1.)]


def test_rdd_zero_edge_coordinate(tmp_path):
    xml, image = example(tmp_path, box=(0, 0, 40, 20))
    assert parse_annotation(xml, image) == [(0, .5, .5, 1., 1.)]


@pytest.mark.parametrize("box,label", [((-1,1,40,20),"D00"), ((1,1,41,20),"D00"), ((20,1,2,20),"D00"), ((1,1,40,20),"OTHER")])
def test_invalid_annotation_rejected(tmp_path, box, label):
    xml, image = example(tmp_path, box=box, label=label)
    with pytest.raises(ValueError):
        parse_annotation(xml, image)


def test_deterministic_disjoint_splits_and_duplicates(tmp_path):
    source = tmp_path / "raw"
    for i in range(10):
        example(source, i)
    # Duplicate pixels with a different filename are removed before splitting.
    example(source / "duplicate", 0)
    first = prepare(source, tmp_path / "out1")
    second = prepare(source, tmp_path / "out2")
    assert first["manifest"] == second["manifest"]
    assert first["duplicates_removed"] == 1
    assert first["images"] == {"train":8,"val":1,"test":1}
    assert len({r["digest"] for r in first["manifest"]}) == 10
    with pytest.raises(ValueError, match="not empty"):
        prepare(source, tmp_path / "out1")


def test_sequence_groups_stay_together(tmp_path):
    source = tmp_path / "raw"
    for i in range(9):
        example(source, i)
    groups = tmp_path / "groups.csv"
    groups.write_text("image,group\n" + "\n".join(f"road{i}.jpg,sequence{i//3}" for i in range(9)))
    result = prepare(source, tmp_path / "out", groups_csv=groups)
    mapping = {}
    for row in result["manifest"]:
        mapping.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in mapping.values())


def test_invalid_ratios_and_empty_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="sum to 1"):
        prepare(raw, tmp_path / "out", ratios=(.7,.2,.2))
    with pytest.raises(ValueError, match="No annotated"):
        prepare(raw, tmp_path / "out")


def test_extra_classes_skip_entire_image_and_report(tmp_path):
    raw = tmp_path / "raw"
    for i in range(4):
        example(raw, i)
    xml, _ = example(raw, 4, label="D44")
    # A mixed image must not have its unsupported object silently turned into background.
    xml.write_text(xml.read_text().replace("</annotation>",
        '<object><name>D40</name><bndbox><xmin>1</xmin><ymin>1</ymin>'
        '<xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>'))
    example(raw, 5, label="D0w0")
    report = prepare(raw, tmp_path / "out")
    assert report["skipped_images"] == 2
    assert report["unsupported_objects"] == {"D44": 1, "D0w0": 1}
    assert sum(report["images"].values()) == 4
    assert all(r["source"] not in {"road4.jpg", "road5.jpg"} for r in report["manifest"])
    persisted = json.loads((tmp_path / "out/split_manifest.json").read_text())
    assert len(persisted["skipped_annotations"]) == 2
    with pytest.raises(ValueError, match="Unsupported classes"):
        prepare(raw, tmp_path / "strict", unsupported_classes="error")


def test_all_images_unsupported_creates_no_output(tmp_path):
    raw = tmp_path / "raw"
    example(raw, label="D44")
    with pytest.raises(ValueError, match="skipped 1 images"):
        prepare(raw, tmp_path / "out")
    assert not (tmp_path / "out").exists()
