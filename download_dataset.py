"""Download the six official N-RDD2024 v3 COCO archives, with SHA256 verification."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def download(item):
    detail = item['content_details']
    path = RAW / item['filename']
    expected = detail['sha256_hash']
    if path.exists():
        if digest(path) == expected:
            print(f'Already verified: {path.name}', flush=True)
            return
        raise ValueError(f'Existing archive has incorrect hash: {path}')
    part = path.with_suffix('.zip.part')
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset != detail['size']:
                req = urllib.request.Request(detail['download_url'], headers={'Range': f'bytes={offset}-', 'User-Agent': 'NRDD2024-research/1.0'})
                with urllib.request.urlopen(req, timeout=90) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected download range')
                    with part.open('ab' if append else 'wb') as f:
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
            if part.stat().st_size != detail['size'] or digest(part) != expected:
                raise ValueError(f'Integrity check failed: {part}')
            part.replace(path)
            print(f'Verified: {path.name} ({detail["size"]:,} bytes)', flush=True)
            return
        except ValueError:
            raise
        except Exception as exc:
            print(f'{path.name}: attempt {attempt + 1}: {exc}', flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'mendeley-files.json').read_text(encoding='utf-8-sig'))
    files = [f for f in manifest if f['filename'].endswith('_coco.zip')]
    if len(files) != 6:
        raise ValueError('Expected six country COCO archives in the version 3 manifest')
    print(f'Downloading {len(files)} archives, {sum(f["size"] for f in files):,} bytes', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))

if __name__ == '__main__':
    main()
