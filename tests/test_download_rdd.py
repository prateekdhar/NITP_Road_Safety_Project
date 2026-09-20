import io
import zipfile
import pytest
from scripts.download_rdd import extract_archive


@pytest.mark.parametrize('nested',[True,False])
def test_extract_country_training_only(tmp_path,nested):
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as outer:
        for country in ['India','Japan','Czech']:
            entries={f'{country}/train/images/a.jpg':b'image',f'{country}/train/annotations/xmls/a.xml':b'xml',f'{country}/test/images/b.jpg':b'test'}
            if nested:
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w') as inner:
                    for name,data in entries.items(): inner.writestr(name,data)
                outer.writestr(f'RDD2022/{country}.zip',buffer.getvalue())
            else:
                for name,data in entries.items(): outer.writestr('RDD2022/'+name,data)
    out=tmp_path/'out'
    assert extract_archive(archive,out,['India','Japan'])=={'India','Japan'}
    assert (out/'RDD2022/Japan/train/images/a.jpg').read_bytes()==b'image'
    assert not (out/'RDD2022/Japan/test').exists()
    assert not (out/'RDD2022/Czech').exists()


def test_reject_traversal(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('India/train/../../../escape.txt','bad')
    with pytest.raises(ValueError,match='Unsafe'):
        extract_archive(archive,tmp_path/'out',['India'])
