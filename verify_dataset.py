"""Independent post-conversion checks before training."""
import json
from collections import defaultdict
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parent

def main():
    folder=ROOT/'data/processed'
    rows=json.loads((folder/'split_manifest.json').read_text())
    identities, contents=defaultdict(set),defaultdict(set)
    seen=set()
    for r in rows:
        identities[r['identity']].add(r['split'])
        contents[r['content']].add(r['split'])
        image=folder/r['output']
        assert image.is_file() and image.stat().st_size, image
        assert r['output'] not in seen, image
        seen.add(r['output'])
        label=folder/'labels'/r['split']/(image.stem+'.txt')
        lines=label.read_text().splitlines()
        for line in filter(str.strip,lines):
            cls,*xywh=map(float,line.split())
            assert cls.is_integer() and 0<=cls<10
            assert len(xywh)==4 and all(math.isfinite(v) and 0<=v<=1 for v in xywh)
            assert xywh[2]>0 and xywh[3]>0
    assert all(len(s)==1 for s in identities.values()), 'Original-image leakage'
    assert all(len(s)==1 for s in contents.values()), 'Identical-pixel leakage'
    print(f'PASS: {len(rows)} image/label pairs; valid labels; no original-ID or identical-pixel leakage.')

if __name__=='__main__':
    main()
