"""Verify every transferred project's file size and critical download/checkpoint hashes."""
import hashlib
import json
import sys
from pathlib import Path

source=Path(__file__).resolve().parent
target=Path(sys.argv[1]).resolve()
checked=0
hashed=0
total=0
errors=[]
for path in source.rglob('*'):
    rel=path.relative_to(source)
    if rel.parts[0] in ('sources','.git') or rel == Path('AGENTS.md') or not path.is_file():
        continue
    other=target/rel
    size=path.stat().st_size
    if not other.is_file() or other.stat().st_size != size:
        errors.append(str(rel))
        continue
    checked+=1
    total+=size
    if path.suffix in ('.zip','.pt'):
        with path.open('rb') as a,other.open('rb') as b:
            if hashlib.file_digest(a,'sha256').digest()!=hashlib.file_digest(b,'sha256').digest():
                errors.append(str(rel))
        hashed+=1
report=dict(files_size_verified=checked,bytes_verified=total,archives_and_checkpoints_sha256_verified=hashed,errors=errors)
print(json.dumps(report,indent=2))
if errors:
    raise SystemExit(1)
