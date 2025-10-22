import json
from pathlib import Path

from tools.build_training_dataset import (
    PROMPT_KEYS,
    build_image_payload,
    choose_unique_output_name,
)


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
        assigned_img_name="sample.webp",
        results_json=results_json,
        quality_labels_for_artist={},
        features_threshold=0.0,
    )

    assert err is None
    assert out_name == "sample.webp"

    prompts = record["prompts"]
    assert set(prompts.keys()) == set(PROMPT_KEYS)
    assert len(record["available_prompts"]) == len(PROMPT_KEYS)

    for key in PROMPT_KEYS:
        value = prompts[key]
        assert value, f"Prompt for {key} is empty"

    glm_withpos = prompts["GLM_NL_WITHPOS"].split("\n")
    assert len(glm_withpos) == 4
    assert glm_withpos[0].startswith(
        "You are an assistant designed to generate anime images based on textual prompts and also to ensure that the characters in the image are positioned according to the normalized bounding box coordinates x1, x2, y1, y2 with values ranging from 1 to 1000. <Prompt Start>"
    )
    assert glm_withpos[1].startswith("Drawn by @")
    assert glm_withpos[2].startswith("Characters:")

    glm_withoutpos = prompts["GLM_NL_WITHOUTPOS"].split("\n")
    assert len(glm_withoutpos) == 3
    assert glm_withoutpos[0].startswith(
        "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start>"
    )
    assert glm_withoutpos[1].startswith("Drawn by @")

    intern_withpos = prompts["InternLM_NL_WITHPOS"].split("\n")
    assert len(intern_withpos) == 4

    intern_withoutpos = prompts["InternLM_NL_WITHOUTPOS"].split("\n")
    assert len(intern_withoutpos) == 3

    glm_xml_lines = prompts["GLM_XML"].split("\n")
    assert len(glm_xml_lines) == 3
    intern_xml_lines = prompts["InternLM_XML"].split("\n")
    assert len(intern_xml_lines) == 3

    dan_prompt = prompts["Danbooru_tags"]
    dan_lines = dan_prompt.split("\n")
    assert len(dan_lines) == 2
    assert "|||" in dan_lines[1]
    artist_tag = record["final_artist_tag"]
    assert "_" in artist_tag
    assert artist_tag in dan_prompt
    cleaned_prompt = dan_prompt.replace(artist_tag, "")
    assert "_" not in cleaned_prompt
    assert "white background" in dan_prompt

def test_choose_unique_output_name_metadata_only(tmp_path):
    dst_artist_dir = tmp_path / "artist"
    dst_artist_dir.mkdir()
    used_names = set()

    name1 = choose_unique_output_name(
        dst_artist_dir=dst_artist_dir,
        base_name="image",
        metadata_only=True,
        original_extension=".png",
        used_names=used_names,
    )
    name2 = choose_unique_output_name(
        dst_artist_dir=dst_artist_dir,
        base_name="image",
        metadata_only=True,
        original_extension=".png",
        used_names=used_names,
    )

    assert name1 == "image.png"
    assert name2 == "image_2.png"
