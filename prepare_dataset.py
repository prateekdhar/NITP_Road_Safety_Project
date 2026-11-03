"""Convert verified country COCO archives to YOLO; audit identity and content leakage."""
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import io
from download_dataset import ROOT, RAW, digest

CODES = [f'D{i}0' for i in range(10)]

def convert_box(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f'Invalid box {box}')
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Box outside image: {box}')
    return [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]

def main():
    out = ROOT / 'data' / 'processed'
    if (out / 'audit.json').exists():
        print('Prepared dataset already exists; preserving it.', flush=True)
        return
    if out.exists():
        raise RuntimeError('Partial processed directory exists; inspect it and choose a new output before retrying')
    files = [f for f in json.loads((ROOT/'mendeley-files.json').read_text(encoding='utf-8-sig')) if f['filename'].endswith('_coco.zip')]
    if len(files) != 6:
        raise ValueError('Expected six COCO archives')
    rows, parents = [], {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        parents[find(a)] = find(b)
    for file in files:
        archive = RAW / file['filename']
        if digest(archive) != file['content_details']['sha256_hash']:
            raise ValueError(f'Archive checksum mismatch: {archive}')
        country = archive.stem.removesuffix('_coco')
        with zipfile.ZipFile(archive) as z:
            for name in sorted(n for n in z.namelist() if n.endswith('.json')):
                doc = json.loads(z.read(name))
                if not {'images','annotations','categories'} <= doc.keys():
                    continue
                cats = {c['id']: c['name'].strip().upper() for c in doc['categories']}
                annotations = defaultdict(list)
                image_ids = {i['id'] for i in doc['images']}
                if len(image_ids) != len(doc['images']):
                    raise ValueError('Duplicate COCO image IDs')
                for a in doc['annotations']:
                    if a['image_id'] not in image_ids or cats[a['category_id']] not in CODES or a.get('iscrowd',0):
                        raise ValueError(f'Unsupported/orphan annotation in {archive}: {a}')
                    annotations[a['image_id']].append(a)
                parts = Path(name).parts
                supplied = 'val' if any(p.lower() in ('val','valid','validation') for p in parts) else 'train'
                if any(p.lower() == 'test' for p in parts):
                    supplied = 'test'
                for im in doc['images']:
                    member = str(Path(name).parent / im['file_name']).replace('\\','/')
                    data = z.read(member)
                    with Image.open(io.BytesIO(data)) as picture:
                        picture.load()
                        if picture.size != (im['width'], im['height']):
                            raise ValueError(f'Incorrect image dimensions: {member}')
                        content = hashlib.sha256(str(picture.size).encode()+picture.convert('RGB').tobytes()).hexdigest()
                    original = re.split(r'(?:_jpg|_png|_jpeg)?\.rf\.', Path(im['file_name']).stem, flags=re.I)[0]
                    identity = country + ':' + original
                    union(identity, 'pixels:'+content)
                    labels = []
                    for a in annotations[im['id']]:
                        box = convert_box(a['bbox'], im['width'], im['height'])
                        labels.append(str(CODES.index(cats[a['category_id']]))+' '+' '.join(f'{v:.8f}' for v in box))
                    rows.append(dict(archive=archive.name, member=member, country=country, identity=identity, content=content, supplied=supplied, labels=sorted(set(labels))))
        print(f'Audited {archive.name}: {len(rows)} cumulative images', flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['identity'])].append(row)
    moved = 0
    for group in groups.values():
        supplied = {r['supplied'] for r in group}
        key = min(r['identity'] for r in group)
        split = 'test' if 'test' in supplied else 'val' if 'val' in supplied else ('test' if int(hashlib.sha256(('42:'+key).encode()).hexdigest()[:8],16) % 10 == 0 else 'train')
        for row in group:
            moved += row['supplied'] == 'train' and split == 'val'
            row['split'] = split
    counts = Counter(r['split'] for r in rows)
    if any(counts[s] == 0 for s in ('train','val','test')):
        raise ValueError(f'Empty split: {counts}')
    archives = {f['filename']:zipfile.ZipFile(RAW/f['filename']) for f in files}
    class_counts = {s:Counter() for s in counts}
    try:
        for row in rows:
            split = row['split']
            ident = hashlib.sha256((row['archive']+'/'+row['member']).encode()).hexdigest()[:20]
            image = out/'images'/split/(ident+Path(row['member']).suffix.lower())
            label = out/'labels'/split/(ident+'.txt')
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(archives[row['archive']].read(row['member']))
            label.write_text('\n'.join(row['labels'])+'\n', encoding='utf-8')
            row['output'] = image.relative_to(out).as_posix()
            class_counts[split].update(CODES[int(line.split()[0])] for line in row['labels'])
    finally:
        for archive in archives.values():
            archive.close()
    for split in counts:
        if set(class_counts[split]) != set(CODES):
            print(f'WARNING: absent classes in {split}: {set(CODES)-set(class_counts[split])}', flush=True)
    (out/'split_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    yaml = 'path: '+json.dumps(out.as_posix())+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {code}\n' for i,code in enumerate(CODES))
    (out/'dataset.yaml').write_text(yaml,encoding='utf-8')
    audit = dict(images=dict(counts), boxes=class_counts, groups=len(groups), training_images_moved_to_validation_to_prevent_leakage=moved, source='10.17632/27c8pwsd6v.3', test='Internal deterministic grouped holdout; not official unlabeled challenge test', limitation='Original-image and identical decoded-pixel grouping; neighboring road frames may still be correlated.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)

if __name__ == '__main__':
    main()
