import torch
import pytest
from road_damage.models import build_model, RoadTrainer


@pytest.mark.parametrize('scale',['n','s','m'])
def test_baseline_scale(scale):
    torch.set_num_threads(2)
    model=build_model(scale=scale,verbose=False)
    model.eval()
    with torch.no_grad():
        prediction=model(torch.randn(1,3,64,64))[0]
    assert prediction.shape==(1,8,84)
    assert torch.isfinite(prediction).all()
    assert model.yaml['scale']==scale


def test_custom_scale_guard():
    with pytest.raises(ValueError,match='custom variants'):
        build_model('pd',scale='s')


def test_trainer_preserves_scale():
    trainer=object.__new__(RoadTrainer)
    trainer.scale='s'
    trainer.variant='baseline'
    trainer.data={'nc':4}
    model=trainer.get_model(verbose=False)
    assert model.yaml['scale']=='s'
