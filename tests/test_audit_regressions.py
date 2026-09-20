from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from road_damage.dataset import parse_annotation, prepare
from road_damage.statistics import read_run
from road_damage.train import validate_data
from PIL import Image


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', ''])
def test_invalid_coordinate_rejected(tmp_path, value):
    image = tmp_path/'image.jpg'
    Image.new('RGB',(20,20)).save(image)
    xml = tmp_path/'image.xml'
    xml.write_text(f'<annotation><object><name>D00</name><bndbox><xmin>{value}</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>')
    with pytest.raises(ValueError):
        parse_annotation(xml,image)


def test_nonfinite_ratio_rejected(tmp_path):
    with pytest.raises(ValueError, match='ratios'):
        prepare(tmp_path, tmp_path.parent/'unused', ratios=(float('nan'),.1,.1))


@pytest.mark.parametrize('content',['','[]','[broken'])
def test_bad_dataset_yaml(tmp_path,content):
    path=tmp_path/'data.yaml'
    path.write_text(content)
    with pytest.raises(ValueError):
        validate_data(path)


def test_nonfinite_log_values(tmp_path):
    (tmp_path/'results.csv').write_text('epoch,metrics/mAP50(B)\nnan,.1\ninf,.2\n1,nan\n')
    assert read_run(tmp_path)['history'] == [{'epoch':1.0}]


def test_real_upload_prediction_from_other_directory(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    weights=root/'runs/train/first_model2/weights/best.pt'
    image=root/'test_samples/pothole_1.jpg'
    if not weights.exists() or not image.exists():
        pytest.skip('Local integration checkpoint and sample are not installed')
    monkeypatch.chdir(tmp_path)
    app=AppTest.from_file(root/'app.py').run(timeout=30)
    assert not app.exception
    app.text_input[0].set_value(str(weights)).run(timeout=30)
    app.get('file_uploader')[0].upload('pothole.jpg',image.read_bytes(),'image/jpeg').run(timeout=30)
    assert not app.exception
    [b for b in app.button if b.label=='Detect road damage'][0].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.get('download_button')) == 2
