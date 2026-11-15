import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from music.director import (
    MusicPlan, apply_instrument_choices, apply_plan_mappings, fallback_plan,
    resolve_music_plan, SYSTEM_PROMPT, MODEL_DEFAULT, summarize_visual_features,
    select_representative_frames,
)
from music.styles import get_style


FEATURES = {
    "timestamps": np.array([0., 1., 2.]), "movement": np.array([0., .5, 1.]),
    "velocity": np.array([0., .5, 1.]), "flow_magnitude": np.array([1., .5, 0.]),
    "activity": np.array([0., .5, 1.]), "smoothness": np.array([1., .5, 0.]),
    "latent_controls": np.array([[-1., 1.], [0., 0.], [1., -1.]]),
}
VALID = {"style": "cinematic", "tempo_bpm": 92, "scale": "major", "mood": "reflective", "explanation": "Broad sustained motion.",
         "mappings": [{"source": "optical_flow", "target": "percussion_density", "strength": .8}],
         "instruments": [{"role": "melody", "timbre": "flute lead"}]}


class MockClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def test_model_valid_json_fences_and_validated_cache(tmp_path):
    client = MockClient(["```json\n" + json.dumps(VALID) + "\n```"])
    cache = tmp_path / "cache"
    result = resolve_music_plan("gentle clouds", FEATURES, client=client, cache_path=cache, api_key="mock")
    assert result.source == "model" and result.plan.style == "cinematic"
    assert MODEL_DEFAULT == "gemma-4-26b-a4b-it"
    assert client.calls[0]["config"]["system_instruction"] == SYSTEM_PROMPT
    assert len(list(cache.glob("*.json"))) == 1
    cached = resolve_music_plan("gentle clouds", FEATURES, client=MockClient([]), cache_path=cache, api_key="mock")
    assert cached.source == "cache" and cached.plan == result.plan
    forced = resolve_music_plan("gentle clouds", FEATURES, client=MockClient([json.dumps(VALID)]),
                                cache_path=cache, force=True, api_key="mock")
    assert forced.source == "model"
    assert "api_key" not in next(cache.glob("*.json")).read_text()
    other = resolve_music_plan("different description", FEATURES, client=MockClient([json.dumps(VALID)]),
                               cache_path=cache, api_key="mock", generation_settings={"seed": 7})
    assert other.cache_key != result.cache_key
    assert len(list(cache.glob("*.json"))) == 2
    settings_variant = resolve_music_plan("gentle clouds", FEATURES, client=MockClient([json.dumps(VALID)]),
                                          cache_path=cache, api_key="mock", generation_settings={"seed": 8})
    assert settings_variant.cache_key != result.cache_key
    assert len(list(cache.glob("*.json"))) == 3


@pytest.mark.parametrize("text", ["not json", json.dumps({**VALID, "style": "bogus"}),
                                    json.dumps({**VALID, "mood": "angry"}),
                                    json.dumps({**VALID, "mappings": [{"source": "rainfall", "target": "melody_contour", "strength": 1}]}),
                                    json.dumps({**VALID, "mappings": [{"source": "optical_flow", "target": "unsupported_control", "strength": 1}]}),
                                    json.dumps({**VALID, "instruments": [{"role": "melody", "timbre": "unregistered"}]}),
                                    json.dumps({**VALID, "extra": "field"})])
def test_invalid_or_malformed_model_output_falls_back(text):
    result = resolve_music_plan("data", FEATURES, client=MockClient([text]), api_key="test")
    assert result.source == "fallback"
    assert isinstance(result.plan, MusicPlan)
    assert "unavailable" in result.warning


def test_api_failure_missing_key_and_offline_fallback():
    failed = resolve_music_plan("x", FEATURES, client=MockClient([RuntimeError("secret response")]), api_key="x", retries=0)
    assert failed.source == "fallback" and "secret response" not in failed.warning
    missing = resolve_music_plan("x", FEATURES, api_key=None, client=MockClient([]))
    offline = resolve_music_plan("x", FEATURES, api_key="x", offline=True)
    assert missing.source == "fallback" and offline.source == "offline"
    assert fallback_plan("jazz").tempo_bpm == get_style("jazz").tempo_bpm


def test_mapping_blends_source_into_composer_override_series():
    no_mapping = apply_plan_mappings(FEATURES, [])
    assert not any(k.startswith("director_") for k in no_mapping)
    result = apply_plan_mappings(FEATURES, [{"source": "latent_z1", "target": "melody_contour", "strength": .5}])
    assert result["timestamps"] is FEATURES["timestamps"]
    # z1 already supplies the target's base control, so blending it with itself preserves bipolar values.
    np.testing.assert_allclose(result["director_melody_contour"], [-1., 0., 1.])
    optical = apply_plan_mappings(FEATURES, [{"source": "optical_flow", "target": "percussion_density", "strength": 1}])
    np.testing.assert_allclose(optical["director_percussion_density"], FEATURES["flow_magnitude"])
    blended = apply_plan_mappings(FEATURES, [{"source": "optical_flow", "target": "dynamic_intensity", "strength": .5}])
    np.testing.assert_allclose(blended["director_dynamic_intensity"], [.5, .5, .5])
    inversion = apply_plan_mappings(FEATURES, [{"source": "visual_activity", "target": "harmony_inversion", "strength": 1}])
    np.testing.assert_allclose(inversion["director_harmony_inversion"], [-1., 0., 1.])
    distinct = {**FEATURES, "velocity": np.array([.2, .4, .6]), "movement": np.array([.8, .6, .4])}
    bases = apply_plan_mappings(distinct, [
        {"source": "optical_flow", "target": "melodic_activity", "strength": 0},
        {"source": "optical_flow", "target": "dynamic_intensity", "strength": 0},
    ])
    np.testing.assert_allclose(bases["director_melodic_activity"], distinct["velocity"])
    np.testing.assert_allclose(bases["director_dynamic_intensity"], distinct["movement"])


def test_summary_describes_dimensionless_controls_and_latents():
    note = summarize_visual_features(FEATURES)["normalization_note"]
    assert "dimensionless" in note and "5th–95th percentile" in note and "physical measurements" in note


def test_resolver_caps_frame_inputs_at_five(tmp_path):
    many = [{"label": f"frame-{i}", "bytes": bytes([i])} for i in range(8)]
    chosen = select_representative_frames(many, limit=5)
    assert len(chosen) == 5
    capped = resolve_music_plan("same", FEATURES, frames=many, cache_path=tmp_path / "cache")
    already_capped = resolve_music_plan("same", FEATURES, frames=chosen, cache_path=tmp_path / "cache")
    assert capped.cache_key == already_capped.cache_key


def test_instrument_choices_only_select_existing_role_specs():
    changed = apply_instrument_choices("jazz", [{"role": "melody", "timbre": "tenor sax"}])
    assert changed[0].role == "melody" and changed[0].gm_program == 66
    assert changed[1:] == get_style("jazz").instruments[1:]
    with pytest.raises(ValueError):
        apply_instrument_choices("cinematic", [{"role": "melody", "timbre": "unregistered"}])
    with pytest.raises(ValueError):
        apply_instrument_choices("jazz", [{"role": "melody", "timbre": "tenor sax"},
                                           {"role": "melody", "timbre": "tenor sax"}])
    with pytest.raises(ValueError):
        MusicPlan(**{**VALID, "instruments": [{"role": "melody", "timbre": "tenor sax"}]})
    with pytest.raises(ValueError):
        MusicPlan(**{**VALID, "instruments": [{"role": "melody", "timbre": "flute lead"},
                                               {"role": "melody", "timbre": "flute lead"}]})


def test_non_retryable_client_status_does_not_retry(monkeypatch):
    class BadRequest(Exception):
        status_code = 400
    client = MockClient([BadRequest("bad request")])
    monkeypatch.setattr("music.director.time.sleep", lambda _: pytest.fail("should not retry"))
    result = resolve_music_plan("x", FEATURES, client=client, api_key="mock", retries=3)
    assert result.source == "fallback"
    assert len(client.calls) == 1


def _install_fake_google_types(monkeypatch, *, client_factory=None):
    google = ModuleType("google")
    google.__path__ = []
    genai = ModuleType("google.genai")
    google_types = ModuleType("google.genai.types")

    class Part:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"bytes": data, "mime_type": mime_type}

    class HttpRetryOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class HttpOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    google_types.Part = Part
    google_types.HttpRetryOptions = HttpRetryOptions
    google_types.HttpOptions = HttpOptions
    genai.types = google_types
    if client_factory is not None:
        genai.Client = client_factory
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", google_types)
    return google_types


def test_image_transient_failure_does_not_start_second_retry_sequence(monkeypatch):
    _install_fake_google_types(monkeypatch)

    class Unavailable(Exception):
        status_code = 503

    client = MockClient([Unavailable("temporary outage")])
    result = resolve_music_plan("storm animation", FEATURES, client=client, api_key="mock",
                                frames=[{"label": "middle", "bytes": b"jpeg"}], retries=0)
    assert result.source == "fallback"
    assert len(client.calls) == 1


def test_permanent_image_rejection_falls_back_to_text_once(monkeypatch):
    _install_fake_google_types(monkeypatch)

    class ImageRejected(Exception):
        status_code = 400

    client = MockClient([ImageRejected("image input is unsupported"), json.dumps(VALID)])
    result = resolve_music_plan("storm animation", FEATURES, client=client, api_key="mock",
                                frames=[{"label": "middle", "bytes": b"jpeg"}], retries=0)
    assert result.source == "model"
    assert not result.image_input_used
    assert "text-only" in result.warning
    assert len(client.calls) == 2
    assert isinstance(client.calls[0]["contents"], list)
    assert isinstance(client.calls[1]["contents"], str)


def test_sdk_retries_are_disabled_in_favor_of_bounded_director_retries(monkeypatch):
    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return MockClient([json.dumps(VALID)])

    _install_fake_google_types(monkeypatch, client_factory=create_client)
    result = resolve_music_plan("gentle clouds", FEATURES, api_key="mock", retries=0, timeout=7.5)
    assert result.source == "model"
    assert captured["http_options"].timeout == 7500
    assert captured["http_options"].retry_options.attempts == 1
