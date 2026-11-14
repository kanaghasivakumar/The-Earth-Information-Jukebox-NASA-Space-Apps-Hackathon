import numpy as np
import pytest, torch, cv2
from PIL import Image
from dataset import (
    NASAVisualizationDataset,
    source_video_split,
    exploratory_time_block_split,
)
from models.vae import ConvVAE, vae_loss


def seq(path, n=12):
    path.mkdir(parents=True)
    for i in range(n):
        a = np.zeros((32, 32, 3), np.uint8)
        a[5:17, 4 + i : 16 + i] = [220, 80, 20]
        Image.fromarray(a).save(path / f"{i:03}.png")


def test_sequence_pair_tensor_sampling_timestamps_and_cache_refresh(tmp_path, monkeypatch):
    src = tmp_path / "seq"
    seq(src)
    cache = tmp_path / "cache"
    ds = NASAVisualizationDataset(src, size=64, fps=1, sequence_fps=2, cache_dir=cache)
    assert (
        len(ds) == 5
    )  # sampled source frames at 0,2,4,6,8,10 seconds; first has no predecessor
    b = ds[0]
    assert b["input"].shape == (5, 64, 64)
    assert torch.equal(b["input"][:3], b["image"])
    assert b["timestamp"].item() == 1
    assert torch.isfinite(b["flow"]).all()
    files = list(cache.glob("*.npz"))
    assert len(files) == 1
    before = ds[0]["image"].clone()
    a = np.full((32, 32, 3), 90, np.uint8)
    Image.fromarray(a).save(src / "010.png")
    ds2 = NASAVisualizationDataset(src, size=64, fps=1, sequence_fps=2, cache_dir=cache)
    assert len(list(cache.glob("*.npz"))) == 2 and not torch.equal(
        before, ds2[-1]["image"]
    )
    # A cache hit decompresses each stored tensor once, rather than once per frame.
    from numpy.lib.npyio import NpzFile
    reads = {}
    original_getitem = NpzFile.__getitem__
    def counted_getitem(archive, key):
        reads[key] = reads.get(key, 0) + 1
        return original_getitem(archive, key)
    monkeypatch.setattr(NpzFile, "__getitem__", counted_getitem)
    ds3 = NASAVisualizationDataset(src, size=64, fps=1, sequence_fps=2, cache_dir=cache)
    assert len(ds3) == 5
    assert reads == {"timestamps": 1, "indices": 1, "rgb": 1, "flow": 1}


def test_gif_timestamps_and_native_video_sampling(tmp_path):
    frames = [Image.fromarray(np.full((32, 32, 3), i * 35, np.uint8)) for i in range(6)]
    gif = tmp_path / "clip.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=200, loop=0)
    ds = NASAVisualizationDataset(gif, size=32, fps=5, cache_dir=tmp_path / "cache")
    assert len(ds) >= 1 and ds[0]["timestamp"].item() == pytest.approx(0.2)


def test_webm_codec_if_available(tmp_path):
    path = tmp_path / "tiny.webm"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP90"), 10, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no WebM VP9 encoder")
    for i in range(20):
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[5:15, 2 + i : 12 + i] = 255
        writer.write(frame)
    writer.release()
    ds = NASAVisualizationDataset(path, size=32, fps=2, cache_dir=tmp_path / "cache")
    assert len(ds) > 1


def test_splits_and_gap(tmp_path):
    seq(tmp_path / "a")
    seq(tmp_path / "b")
    ds = NASAVisualizationDataset(
        tmp_path, size=32, sequence_fps=5, cache_dir=tmp_path / "cache"
    )
    tr, va = source_video_split(ds, seed=5)
    assert {ds.samples[i][0] for i in tr}.isdisjoint({ds.samples[i][0] for i in va})
    tr, va = exploratory_time_block_split(ds, source_id=0, gap=2)
    assert max(tr) < min(va) and len(tr) + len(va) < sum(
        sid == 0 for sid, *_ in ds.samples
    )


def test_model_contract_and_loss():
    m = ConvVAE(3)
    x = torch.rand(2, 5, 64, 64)
    o = m(x, True)
    assert o["rgb"].shape == (2, 3, 64, 64) and o["flow"].shape == (2, 2, 64, 64)
    assert torch.equal(o["z"], o["mu"])
    small = ConvVAE(2, image_size=32)
    assert small(torch.rand(1, 5, 32, 32), True)["rgb"].shape == (1, 3, 32, 32)
    assert torch.equal(m.reparameterize(o["mu"], o["logvar"], True), o["mu"])
    loss, parts = vae_loss(o, x[:, :3], x[:, 3:], beta=0.01)
    assert torch.isfinite(loss) and parts["kl"].ndim == 0
    with pytest.raises(ValueError):
        ConvVAE(4)

def test_evaluation_accepts_legacy_and_current_image_size_keys():
 from evaluate import checkpoint_image_size
 assert checkpoint_image_size({'model_config':{'size':32}})==32
 assert checkpoint_image_size({'model_config':{'image_size':64}})==64
 with pytest.raises(ValueError): checkpoint_image_size({'model_config':{'latent_dim':2}})


def test_experimental_profile_defaults_and_kl_warmup():
    from train import (
        DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_FPS, DEFAULT_FLOW_SCALE,
        DEFAULT_FLOW_WEIGHT, DEFAULT_BETA, DEFAULT_KL_WARMUP_EPOCHS, warmup_beta,
    )
    assert (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_FPS) == (50, 8, 5.0)
    assert (DEFAULT_FLOW_SCALE, DEFAULT_FLOW_WEIGHT, DEFAULT_BETA, DEFAULT_KL_WARMUP_EPOCHS) == (1.0, 0.3, 0.01, 10)
    assert warmup_beta(0.01, 10, 0) == 0
    assert warmup_beta(0.01, 10, 1) == pytest.approx(0.001)
    assert warmup_beta(0.01, 10, 10) == pytest.approx(0.01)
    assert warmup_beta(0.01, 10, 20) == pytest.approx(0.01)
    assert warmup_beta(0.01, 0, 1) == pytest.approx(0.01)
