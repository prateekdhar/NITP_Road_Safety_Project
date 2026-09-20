"""Read-only audit of all prepared images, labels and their source annotations."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from PIL import Image
from road_damage import CLASSES
from road_damage.dataset import parse_annotation


def main():
    root = Path(__file__).resolve().parents[1]
    data = root/'data/rdd'
    manifest = json.loads((data/'split_manifest.json').read_text())
    entries = {row['digest'][:24]:row for row in manifest['manifest']}
    seen = set()
    counts, objects, errors = {}, {}, []
    for split in ('train','val','test'):
        counter = Counter()
        images = sorted((data/'images'/split).glob('*'))
        counts[split] = len(images)
        if {p.stem for p in images} != {p.stem for p in (data/'labels'/split).glob('*.txt')}:
            errors.append(f'{split}: image/label pairs differ')
        for image in images:
            try:
                entry = entries[image.stem]
                assert entry['split'] == split, 'manifest split mismatch'
                with Image.open(image) as im:
                    rgb = im.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                assert digest == entry['digest'], 'image digest mismatch'
                assert digest not in seen, 'duplicate pixels across prepared images'
                seen.add(digest)
                rows = []
                for line in (data/'labels'/split/(image.stem+'.txt')).read_text().splitlines():
                    values = line.split()
                    assert len(values) == 5, 'label must have five fields'
                    cls = int(values[0])
                    x,y,w,h = map(float, values[1:])
                    assert cls in CLASSES and all(math.isfinite(v) for v in (x,y,w,h)), 'invalid class or coordinate'
                    assert 0 < w <= 1 and 0 < h <= 1, 'invalid box size'
                    assert x-w/2 >= -1e-7 and y-h/2 >= -1e-7 and x+w/2 <= 1+1e-7 and y+h/2 <= 1+1e-7, 'box out of bounds'
                    rows.append((cls,x,y,w,h))
                    counter[CLASSES[cls]] += 1
                source = root/'RDD2022'/entry['source']
                xml = source.parent.parent/'annotations/xmls'/f'{source.stem}.xml'
                assert source.is_file() and xml.is_file(), 'source image or annotation missing'
                expected = parse_annotation(xml, source)
                assert len(rows) == len(expected), 'source label count differs'
                assert all(a[0] == b[0] and all(abs(v-t) < 1e-7 for v,t in zip(a[1:],b[1:])) for a,b in zip(rows,expected)), 'converted labels differ from source'
            except Exception as exc:
                errors.append(f'{image.name}: {exc}')
        objects[split] = {c:counter[c] for c in CLASSES.values()}
    if counts != manifest['images'] or objects != manifest['objects']:
        errors.append('manifest counts differ from disk')
    if len(seen) != len(entries):
        errors.append('manifest membership differs from disk')
    report = {'images':counts, 'objects':objects, 'verified_unique_images':len(seen), 'errors':errors}
    out = root/'docs/dataset_audit.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
