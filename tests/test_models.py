import pytest
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from road_damage.models import build_model, VARIANTS
from road_damage.inference import load_detector
from road_damage import CLASSES

torch.set_num_threads(2)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_models_forward_backward_and_decode(variant):
    model = build_model(variant, verbose=False)
    model.train()
    predictions = model(torch.randn(2,3,64,64))
    assert len(predictions) == 3
    assert [p.shape[-1] for p in predictions] == [8,4,2]
    model.args = get_cfg()
    loss, components = model.loss({
        'img': torch.randn(2,3,64,64), 'batch_idx': torch.tensor([0.,1.]),
        'cls': torch.tensor([[0.],[3.]]),
        'bboxes': torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]]),
    }, preds=predictions)
    loss = loss.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(components).all()
    model.eval()
    with torch.no_grad():
        decoded, _ = model(torch.randn(1,3,64,64))
    assert decoded.shape == (1,8,84)
    assert torch.isfinite(decoded).all()


def test_custom_checkpoint_roundtrip(tmp_path):
    detector = YOLO("yolov8n.yaml")
    detector.model = build_model("pd", verbose=False)
    detector.model.names = CLASSES
    path = tmp_path / "custom.pt"
    detector.save(path)
    restored = load_detector(path)
    assert restored.model.variant == "pd"
    assert restored.names == CLASSES


def test_coco_checkpoint_rejected(tmp_path):
    model = YOLO("yolov8n.yaml")
    path = tmp_path / "coco.pt"
    model.save(path)
    with pytest.raises(ValueError, match="four-class"):
        load_detector(path)
