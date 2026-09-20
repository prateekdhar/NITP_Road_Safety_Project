"""Download the official RDD2022 archive and extract annotated country subsets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile
import requests


def extract_archive(archive, destination, countries):
    """Handle the official outer ZIP of country ZIPs, or a flat archive."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    found = set()

    def extract_training(z, country):
        count = 0
        for item in z.infolist():
            parts = PurePosixPath(item.filename.replace('\\', '/')).parts
            if 'train' not in parts or item.is_dir():
                continue
            if '..' in parts or PurePosixPath(item.filename).is_absolute():
                raise ValueError(f'Unsafe archive path: {item.filename}')
            relative = Path(*parts[parts.index('train'):])
            base = destination/'RDD2022'/country
            target = (base/relative).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f'Unsafe archive path: {item.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
            count += 1
        if count:
            found.add(country)
        print(f'{country}: extracted {count} training files', flush=True)

    with zipfile.ZipFile(archive) as outer:
        for country in countries:
            nested = [item for item in outer.infolist()
                      if PurePosixPath(item.filename.replace('\\','/')).name == f'{country}.zip']
            if nested:
                if len(nested) != 1:
                    raise ValueError(f'Ambiguous nested archive for {country}')
                print(f'Unpacking {country}.zip...', flush=True)
                with tempfile.TemporaryDirectory(prefix='rdd-extract-', dir=destination) as work:
                    path = Path(work)/'country.zip'
                    with outer.open(nested[0]) as source, path.open('wb') as output:
                        shutil.copyfileobj(source, output, 1024*1024)
                    with zipfile.ZipFile(path) as inner:
                        extract_training(inner, country)
            else:
                # Restrict the flat archive view to the selected country's members.
                class CountryView:
                    def infolist(self):
                        return [i for i in outer.infolist() if country in PurePosixPath(i.filename.replace('\\','/')).parts]
                    def open(self, item):
                        return outer.open(item)
                extract_training(CountryView(), country)
    if found != set(countries):
        raise ValueError(f'Missing requested countries: {set(countries)-found}')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--countries', nargs='+', default=['India','Japan','Czech'])
    parser.add_argument('--output', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    response = requests.get('https://api.figshare.com/v2/articles/21431547',timeout=60)
    response.raise_for_status()
    metadata = response.json()
    file = next(f for f in metadata['files'] if f['name']=='RDD2022_released_through_CRDDC2022.zip')
    archive = args.output/file['name']
    if not archive.is_file():
        partial = archive.with_suffix('.zip.partial')
        # Restart incomplete downloads rather than combine responses from different versions.
        with requests.get(file['download_url'],stream=True,timeout=(30,120)) as stream:
            stream.raise_for_status()
            downloaded = 0
            with partial.open('wb') as target:
                for chunk in stream.iter_content(8*1024*1024):
                    target.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (256*1024*1024) < len(chunk):
                        print(f'{downloaded/1e9:.2f} / {file["size"]/1e9:.2f} GB',flush=True)
        partial.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream,'md5').hexdigest()
    if archive.stat().st_size != file['size'] or digest != file['computed_md5']:
        raise ValueError('Archive integrity mismatch; remove the downloaded archive and retry')
    print('Archive checksum verified. Extracting country training subsets...', flush=True)
    found = extract_archive(archive, args.output, args.countries)
    (args.output/'download_source.json').write_text(json.dumps({'article':metadata['id'],'file':file,'countries':sorted(found)},indent=2))
    print('Verified archive and extracted annotated training subsets:',sorted(found))


if __name__=='__main__':
    main()
