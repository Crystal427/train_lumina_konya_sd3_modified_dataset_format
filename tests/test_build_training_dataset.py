import json
from pathlib import Path

from tools.build_training_dataset import PROMPT_KEYS, build_image_payload


def test_build_image_payload_prompts_cover_all_keys(tmp_path):
    artist_name = "artist_sample_name"
    artist_dir = tmp_path / artist_name
    year_dir = artist_dir / "new"
    year_dir.mkdir(parents=True, exist_ok=True)

    dst_artist_dir = tmp_path / "out" / artist_name
    dst_artist_dir.mkdir(parents=True, exist_ok=True)

    nl_root = tmp_path / "nl"
    glm_dir = nl_root / artist_name / "NL_caption_GLM"
    glm_dir.mkdir(parents=True, exist_ok=True)
    intern_dir = nl_root / artist_name / "NLcaption"
    intern_dir.mkdir(parents=True, exist_ok=True)

    image_path = year_dir / "sample.png"
    image_path.touch()

    glm_payload = {
        "character_1": {
            "bbox": [10, 20, 30, 40],
            "name": "$character_1$",
        },
        "image": {
            "caption": "A scene with $character_1$ tying her hair.",
            "tags": "<root>$character_1$ appears here</root>",
        },
    }
    intern_payload = {
        "image": {
            "caption": "Another view featuring $character_1$.",
            "tags": "<scene>$character_1$</scene>",
        }
    }

    glm_path = glm_dir / "sample.json"
    intern_path = intern_dir / "sample.json"
    glm_path.write_text(json.dumps(glm_payload), encoding="utf-8")
    intern_path.write_text(json.dumps(intern_payload), encoding="utf-8")

    results_json = {
        "sample.png": {
            "imgscore": {
                "illustration": 0.9,
                "comic": 0.05,
                "not_painting": 0.03,
            },
            "features": {
                "1girl": 0.95,
                "long_hair": 0.88,
                "white_background": 0.75,
                "simple_background": 0.6,
            },
            "character": {"heroine": 0.8},
            "rating": {"general": 0.95},
            "additional_tags": "blue_sky",
        }
    }

    record, out_name, err = build_image_payload(
        nl_root=nl_root,
        artist_dir=artist_dir,
        dst_artist_dir=dst_artist_dir,
        year_name="new",
        img_path=image_path,
        image_index=1,
        results_json=results_json,
        quality_labels_for_artist={},
        features_threshold=0.0,
    )

    assert err is None
    assert out_name.endswith(".webp")

    prompts = record["prompts"]
    assert set(prompts.keys()) == set(PROMPT_KEYS)
    assert len(record["available_prompts"]) == len(PROMPT_KEYS)

    for key in PROMPT_KEYS:
        value = prompts[key]
        assert value, f"Prompt for {key} is empty"
        assert "\n" not in value, f"Prompt for {key} contains newline characters"

    dan_prompt = prompts["Danbooru_tags"]
    artist_tag = record["final_artist_tag"]
    assert "_" in artist_tag
    assert artist_tag in dan_prompt
    cleaned_prompt = dan_prompt.replace(artist_tag, "")
    assert "_" not in cleaned_prompt
    assert "white background" in dan_prompt
