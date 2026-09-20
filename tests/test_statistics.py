from pathlib import Path

from streamlit.testing.v1 import AppTest

from road_damage.statistics import read_run


def test_log_reader_missing_and_partial_values(tmp_path):
    assert read_run(tmp_path)['history'] == []
    (tmp_path/'results.csv').write_text(' epoch , metrics/mAP50(B),metrics/recall(B)\n1,0.08328,0.12303\n2,,\n')
    report = read_run(tmp_path)
    assert report['history'][0]['metrics/mAP50(B)'] == .08328
    assert 'metrics/mAP50(B)' not in report['history'][1]
    assert report['checkpoint_exists'] is False


def test_statistics_page_and_checkpoint_inspection():
    path = Path(__file__).resolve().parents[1]/'app.py'
    app = AppTest.from_file(path).run(timeout=30)
    app.switch_page('pages/1_Model_Statistics.py').run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'Model statistics & characteristics'
    assert any(m.value == '70.6%' for m in app.metric)
    if app.selectbox:
        assert 'smoke_pd' not in app.selectbox[0].options
        buttons = [b for b in app.button if b.label == 'Inspect checkpoint characteristics']
        if buttons:
            buttons[0].click().run(timeout=30)
            assert not app.exception
            assert any(m.label == 'Parameters in saved checkpoint' for m in app.metric)
