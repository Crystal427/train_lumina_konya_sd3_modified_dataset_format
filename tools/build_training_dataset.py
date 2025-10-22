import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm


# ----------------------------
# Robust image handling setup
# ----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ----------------------------
# Constants and configuration
# ----------------------------
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
YEAR_FOLDERS = ["2010s", "2017s", "2020s", "2022s", "new", "undefined", "unknown"]
YEAR_PRIORITY = ["new", "2022s", "2020s", "2017s", "2010s", "undefined", "unknown"]
SPECIAL_FILTER_YEARS = {"new", "2022s"}
FILTER_FEATURE_KEYS = {
    "monochrome",
    "simple_background",
    "white_background",
    "transparent_background",
}
FEATURES_THRESHOLD_DEFAULT = 0.27


# ----------------------------
# Utility helpers
# ----------------------------
def safe_read_json(json_path: Path) -> Optional[dict]:
    try:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def normalized_join(*parts: str) -> str:
    return str(Path(*parts))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_artist_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def splitext_lower(name: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(name)
    return base, ext.lower()


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def get_results_entry(
    results_json: Optional[dict],
    year_name: str,
    img_path: Path,
) -> Optional[dict]:
    if not results_json:
        return None
    base_name = img_path.name
    candidates = [base_name]

    def add_variant(prefix: Optional[str]) -> None:
        if not prefix:
            return
        candidates.append(f"{prefix}/{base_name}")
        candidates.append(f"{prefix}\\{base_name}")

    add_variant(year_name)
    if year_name == "new" and any(part.lower() == "augmentation" for part in img_path.parts):
        add_variant("Augmentation")
        add_variant(os.path.join("Augmentation", year_name))

    for key in candidates:
        if key in results_json:
            entry = results_json.get(key)
            if isinstance(entry, dict):
                return entry
    return None


def should_defer_image(year_name: str, img_path: Path, results_json: Optional[dict]) -> bool:
    if year_name not in SPECIAL_FILTER_YEARS:
        return False
    entry = get_results_entry(results_json, year_name, img_path)
    if not entry:
        return False

    imgscore = entry.get("imgscore")
    top_key = None
    if isinstance(imgscore, dict):
        numeric_scores = {}
        for key, value in imgscore.items():
            try:
                numeric_scores[key] = float(value)
            except (TypeError, ValueError):
                continue
        if numeric_scores:
            top_key = max(numeric_scores, key=numeric_scores.get)
    if top_key in {"comic", "not_painting"}:
        return True

    features = entry.get("features")
    if isinstance(features, dict):
        for tag in FILTER_FEATURE_KEYS:
            value = features.get(tag)
            try:
                if value is not None and float(value) > 0.4:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def order_artist_images(
    images: List[Tuple[str, Path]],
    results_json: Optional[dict],
) -> List[Tuple[str, Path]]:
    new_regular: List[Tuple[str, Path]] = []
    new_filtered: List[Tuple[str, Path]] = []
    year2022_regular: List[Tuple[str, Path]] = []
    year2022_filtered: List[Tuple[str, Path]] = []
    grouped: Dict[str, List[Tuple[str, Path]]] = {year: [] for year in YEAR_PRIORITY[2:]}
    others: List[Tuple[str, Path]] = []

    def sort_group(items: List[Tuple[str, Path]]) -> None:
        items.sort(key=lambda pair: pair[1].name.lower())

    for year_name, path in images:
        filtered = should_defer_image(year_name, path, results_json)
        if year_name == "new":
            (new_filtered if filtered else new_regular).append((year_name, path))
        elif year_name == "2022s":
            (year2022_filtered if filtered else year2022_regular).append((year_name, path))
        elif year_name in grouped:
            grouped[year_name].append((year_name, path))
        else:
            others.append((year_name, path))

    sort_group(new_regular)
    sort_group(new_filtered)
    sort_group(year2022_regular)
    sort_group(year2022_filtered)
    for year_list in grouped.values():
        sort_group(year_list)
    others.sort(key=lambda pair: (pair[0], pair[1].name.lower()))

    ordered: List[Tuple[str, Path]] = []
    ordered.extend(new_regular)
    ordered.extend(year2022_regular)
    ordered.extend(new_filtered)
    ordered.extend(year2022_filtered)
    for year in YEAR_PRIORITY[2:]:
        ordered.extend(grouped.get(year, []))
    ordered.extend(others)
    return ordered


def try_exif_transpose(img: Image.Image) -> Image.Image:
    try:
        return Image.Image.transpose(img, Image.Transpose.EXIF)
    except Exception:
        return img


WHITESPACE_RE = re.compile(r"\s+")


def strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def collapse_whitespace(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = WHITESPACE_RE.sub(" ", str(value))
    text = text.strip()
    return text if text else None


def compose_prompt(header: str, segments: List[Optional[str]]) -> Optional[str]:
    if not header:
        return None
    lines: List[str] = [collapse_whitespace(header) or header]
    for segment in segments:
        cleaned = collapse_whitespace(segment)
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return None
    return "\n".join(lines)


def extract_model_field(model_json: Optional[dict], field: str) -> Optional[str]:
    if not model_json or not isinstance(model_json.get("image"), dict):
        return None
    return strip_or_none(model_json.get("image", {}).get(field))


GLM_NL_WITHPOS_HEADER = (
    "You are an assistant designed to generate anime images based on textual prompts and also to ensure that the characters in the image are positioned according to the normalized bounding box coordinates x1, x2, y1, y2 with values ranging from 1 to 1000. <Prompt Start>"
)
GLM_NL_WITHOUTPOS_HEADER = (
    "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start>"
)
GLM_XML_HEADER = (
    "You are an assistant designed to generate anime images based on structured XML descriptions. <Prompt Start>"
)
INTERNLM_NL_WITHPOS_HEADER = GLM_NL_WITHPOS_HEADER
INTERNLM_NL_WITHOUTPOS_HEADER = GLM_NL_WITHOUTPOS_HEADER
INTERNLM_XML_HEADER = GLM_XML_HEADER
DANBOORU_HEADER = (
    "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags. <Prompt Start>"
)

PROMPT_KEYS = [
    "GLM_NL_WITHPOS",
    "GLM_NL_WITHOUTPOS",
    "GLM_XML",
    "InternLM_NL_WITHPOS",
    "InternLM_NL_WITHOUTPOS",
    "InternLM_XML",
    "Danbooru_tags",
]


# ----------------------------
# Danbooru tag processing (reference-aligned)
# ----------------------------
PATTERN_ESCAPED_BRACKET = r"\\([\(\)\[\]\{\}])"

def unescape_brackets(s: str) -> str:
    return re.sub(PATTERN_ESCAPED_BRACKET, r"\1", s)


def fmt2danbooru(tag: str) -> str:
    tag = tag.lower().replace(" ", "_").strip("_").replace(":_", ":")
    tag = unescape_brackets(tag)
    return tag


def get_year_from_weibo_date(date_str: str) -> Optional[int]:
    try:
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt.year
    except Exception:
        return None


def get_year_from_date(date_str: str) -> Optional[int]:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.year
    except Exception:
        return None


@dataclass
class DanbooruComponents:
    final_features_tag_prefix: str
    final_features_tag: str
    final_character_tag: str
    final_copyright_tag: str
    year_tag_specific: str
    year_tag: str
    final_rating_tag: str
    additional_tags: str
    aes_rating: str


def analyze_image_for_danbooru(
    artist_folder: Path,
    year_folder_name: str,
    filename: str,
    results_json: Optional[dict],
    quality_labels_for_artist: Optional[Dict[str, str]],
    features_threshold: float,
) -> DanbooruComponents:
    name, _ = os.path.splitext(filename)

    # Resolve source path (handle new/Augmentation)
    if year_folder_name == "new" and "Augmentation" + os.sep in filename:
        relative_no_aug = filename.replace("Augmentation" + os.sep, "")
        src_img_path = artist_folder / year_folder_name / "Augmentation" / relative_no_aug
        filename_without_aug = relative_no_aug
    else:
        src_img_path = artist_folder / year_folder_name / filename
        filename_without_aug = filename

    jsons_folder = artist_folder / "jsons"

    # Optional sidecar text overrides
    txt_path = src_img_path.with_suffix(".txt")
    finaltag_dan_from_txt: Optional[str] = None
    if txt_path.exists():
        try:
            finaltag_dan_from_txt = txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            finaltag_dan_from_txt = None

    # Load danbooru image json (prefix by first 16 chars of name; allow ext variations)
    danbooru_json = None
    if jsons_folder.exists():
        try:
            name_prefix = name[:16]
            for json_file in os.listdir(jsons_folder):
                if not json_file.endswith((".json", ".png.json", ".jpg.json", ".jpeg.json", ".webp.json")):
                    continue
                if json_file.startswith(name_prefix):
                    candidate = jsons_folder / json_file
                    danbooru_json = safe_read_json(candidate)
                    if danbooru_json is not None:
                        break
        except Exception:
            danbooru_json = None

    final_character_tag = ""
    if danbooru_json and isinstance(danbooru_json.get("tags_character"), list) and danbooru_json["tags_character"]:
        final_character_tag = ", ".join(danbooru_json["tags_character"]).replace("_", " ")
    elif (
        results_json
        and filename_without_aug in results_json
        and isinstance(results_json[filename_without_aug].get("character"), dict)
        and results_json[filename_without_aug]["character"]
    ):
        final_character_tag = ", ".join(results_json[filename_without_aug]["character"].keys()).replace("_", " ")
    final_character_tag = final_character_tag.strip(", ")

    final_copyright_tag = ""
    if danbooru_json and isinstance(danbooru_json.get("tags_copyright"), list):
        filtered_tags = [t for t in danbooru_json["tags_copyright"] if str(t).lower() != "original"]
        final_copyright_tag = ",".join(filtered_tags).replace("_", " ").strip(", ")

    # features: mix of danbooru_json tags_general and results_json features with threshold
    features_tag: set[str] = set()
    prefix_features_tag: set[str] = set()
    person_count_patterns = [
        re.compile(r"^(\d+|6\+)(girls?|boys?|others?)$", re.IGNORECASE),
        re.compile(r"^multiple_(girls|boys|others)$", re.IGNORECASE),
    ]

    def is_person_count(tag_text: str) -> bool:
        for pat in person_count_patterns:
            if pat.match(tag_text):
                return True
        return False

    if danbooru_json and isinstance(danbooru_json.get("tags_general"), list):
        for raw_tag in danbooru_json["tags_general"]:
            tag_text = str(raw_tag).replace("_", " ")
            (prefix_features_tag if is_person_count(tag_text) else features_tag).add(tag_text)

    if results_json and filename_without_aug in results_json and isinstance(
        results_json[filename_without_aug].get("features"), dict
    ):
        for k, v in results_json[filename_without_aug]["features"].items():
            try:
                score = float(v)
            except Exception:
                continue
            if score > features_threshold:
                tag_text = str(k).replace("_", " ")
                (prefix_features_tag if is_person_count(tag_text) else features_tag).add(tag_text)

    final_features_tag = ", ".join(sorted(features_tag))
    final_features_tag_prefix = ", ".join(sorted(prefix_features_tag))

    # Comment (native) — not required for this task beyond being part of native caption; we ignore in outputs
    # Rating, resolution-based tags, AI flag
    final_rating_tag = ""
    max_rating = "general"
    if results_json and filename_without_aug in results_json:
        data = results_json[filename_without_aug]
        if data.get("is_AI"):
            final_rating_tag += "ai-generated, "
        rating_dict = data.get("rating") or {}
        if isinstance(rating_dict, dict) and rating_dict:
            try:
                max_rating = max(rating_dict, key=rating_dict.get)
            except Exception:
                max_rating = "general"
            mapped = "safe"
            if max_rating == "questionable":
                mapped = "nsfw"
            elif max_rating in {"explicit", "nsfw"}:
                mapped = "explicit" if max_rating == "explicit" else "nsfw"
            final_rating_tag += f"{mapped}, "

        try:
            with Image.open(src_img_path) as img:
                width, height = img.size
                area = width * height
                if area <= 589_824:
                    final_rating_tag += "lowres, "
                elif area >= 1_638_400:
                    final_rating_tag += "absurdres, "
        except Exception:
            pass

    aes_rating = ""
    if quality_labels_for_artist and filename_without_aug in quality_labels_for_artist:
        aes_rating = quality_labels_for_artist.get(filename_without_aug) or ""
        if aes_rating:
            final_rating_tag += f"{aes_rating}, "

    additional_tags = ""
    if results_json and filename_without_aug in results_json:
        add_tags = results_json[filename_without_aug].get("additional_tags")
        if isinstance(add_tags, str):
            additional_tags = add_tags.replace("_", " ")

    # Year-level tags
    year_tag = ""
    if year_folder_name in {"new", "2022s", "2020s", "2017s", "2010s"}:
        mapping = {
            "new": "newest, ",
            "2022s": "recent, ",
            "2020s": "mid, ",
            "2017s": "early, ",
            "2010s": "oldest, ",
        }
        year_tag = mapping.get(year_folder_name, "")

    year_tag_specific = ""
    if year_folder_name in {"2020s", "2022s", "new"}:
        json_path_exact = (artist_folder / "jsons" / f"{name}.json")
        year_val: Optional[int] = None
        data = safe_read_json(json_path_exact)
        if data:
            try:
                if data.get("category") == "weibo":
                    year_val = get_year_from_weibo_date(data.get("status", {}).get("created_at", ""))
                else:
                    year_val = get_year_from_date(data.get("date", ""))
            except Exception:
                year_val = None
        if year_val is None and src_img_path.exists():
            try:
                mtime = src_img_path.stat().st_mtime
                year_val = datetime.fromtimestamp(mtime).year
            except Exception:
                year_val = None
        if year_val:
            if year_folder_name == "2020s":
                year_val = max(2018, min(year_val, 2020))
            elif year_folder_name == "2022s":
                year_val = max(2021, min(year_val, 2022))
            elif year_folder_name == "new":
                year_val = max(2023, min(year_val, 2025))
            year_tag_specific = f"year {year_val}, "

    # Normalize trailing commas and spaces on string components
    def strip_trailing_commas(s: str) -> str:
        return s.strip().strip(",").strip()

    final_features_tag_prefix = strip_trailing_commas(final_features_tag_prefix)
    final_features_tag = strip_trailing_commas(final_features_tag)
    final_character_tag = strip_trailing_commas(final_character_tag)
    final_copyright_tag = strip_trailing_commas(final_copyright_tag)
    year_tag_specific = strip_trailing_commas(year_tag_specific)
    year_tag = strip_trailing_commas(year_tag)
    final_rating_tag = strip_trailing_commas(final_rating_tag)
    additional_tags = strip_trailing_commas(additional_tags)

    return DanbooruComponents(
        final_features_tag_prefix=final_features_tag_prefix,
        final_features_tag=final_features_tag,
        final_character_tag=final_character_tag,
        final_copyright_tag=final_copyright_tag,
        year_tag_specific=year_tag_specific,
        year_tag=year_tag,
        final_rating_tag=final_rating_tag,
        additional_tags=additional_tags,
        aes_rating=aes_rating,
    )


def compose_danbooru_metadata(
    components: DanbooruComponents,
    final_artist_tag: Optional[str],
) -> Dict[str, Optional[str]]:
    def join_nonempty(parts: List[str]) -> str:
        cleaned = [p.strip() for p in parts if p and p.strip()]
        return ", ".join(cleaned)

    prefix_parts = [
        components.final_features_tag_prefix,
        final_artist_tag or "",
        components.final_character_tag,
        components.final_copyright_tag,
        components.year_tag_specific,
    ]
    suffix_parts = [
        components.final_features_tag,
        components.year_tag,
        components.final_rating_tag,
        components.additional_tags,
    ]

    prefix = join_nonempty(prefix_parts)
    suffix = join_nonempty(suffix_parts)
    finaltag_dan = join_nonempty([prefix, suffix])

    if prefix and suffix:
        prompt_body = f"{prefix},|||{suffix}"
    elif prefix:
        prompt_body = prefix
    elif suffix:
        prompt_body = suffix
    else:
        prompt_body = ""

    return {
        "prefix": prefix or None,
        "suffix": suffix or None,
        "finaltag_dan": finaltag_dan or None,
        "prompt_body": prompt_body or None,
    }


# ----------------------------
# GLM & InternLM parsing
# ----------------------------
def parse_glm_characters(glm_json: dict) -> Tuple[List[Tuple[str, List[int]]], Dict[str, str]]:
    """
    Returns:
        - ordered list of (canonical_name_without_hash, bbox_as_[x1,x2,y1,y2])
        - mapping from json placeholder tokens (e.g. "$character_1$") to replacement text (e.g. "character_1" or "#miku")
    """
    if not glm_json:
        return [], {}

    character_entries: List[Tuple[str, List[int]]] = []
    placeholder_to_label: Dict[str, str] = {}

    # detect all character_* keys except "image"
    candidates = [(k, v) for k, v in glm_json.items() if k != "image" and isinstance(v, dict)]

    def to_canonical_name(raw_name: str) -> str:
        if raw_name.startswith("$") and raw_name.endswith("$") and len(raw_name) >= 3:
            inner = raw_name[1:-1]
        else:
            inner = raw_name
        return inner

    def order_key(name: str) -> Tuple[int, str]:
        m = re.match(r"character_(\d+)$", name)
        if m:
            return int(m.group(1)), name
        return (10**9, name)

    for key, entry in candidates:
        bbox = entry.get("bbox")
        name_raw = entry.get("name", f"${key}$")
        try:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            # Assume bbox in [x1, y1, x2, y2] and convert to [x1, x2, y1, y2]
            x1, y1, x2, y2 = bbox
            x1 = clamp_int(int(round(float(x1))), 1, 1000)
            x2 = clamp_int(int(round(float(x2))), 1, 1000)
            y1 = clamp_int(int(round(float(y1))), 1, 1000)
            y2 = clamp_int(int(round(float(y2))), 1, 1000)
            bbox_conv = [x1, x2, y1, y2]
        except Exception:
            continue

        canonical = to_canonical_name(str(name_raw))
        character_entries.append((canonical, bbox_conv))

    # sort characters: character_1, character_2, ..., then others lexicographically
    character_entries.sort(key=lambda t: order_key(t[0]))

    for canonical, _ in character_entries:
        placeholder = f"${canonical}$"
        if re.match(r"character_\d+$", canonical):
            replacement = canonical  # without '#', spec asks plain character_n in caption
            label = f"#{canonical}"
        else:
            replacement = canonical
            label = f"#{canonical}"
        placeholder_to_label[placeholder] = label

    return character_entries, placeholder_to_label


def build_characters_line(character_entries: List[Tuple[str, List[int]]]) -> str:
    if not character_entries:
        return ""
    segments = []
    for canonical, bbox in character_entries:
        label = f"#{canonical}"
        x1, x2, y1, y2 = bbox
        segments.append(f"{label}:[{x1},{x2},{y1},{y2}]")
    return "Characters: " + ",".join(segments)


def apply_placeholder_replacements(text: str, placeholder_to_label: Dict[str, str]) -> str:
    if not text:
        return text
    out = text
    # Replace $name$ with corresponding label or plain canonical
    for placeholder, label in placeholder_to_label.items():
        canonical = placeholder.strip("$")
        # In caption we need plain canonical (without '#') for readability as per spec
        if re.match(r"character_\d+$", canonical):
            repl_in_caption = canonical
        else:
            repl_in_caption = label  # use #name for real names
        out = out.replace(placeholder, repl_in_caption)
    return out


def build_characters_payload(
    character_entries: List[Tuple[str, List[int]]], placeholder_to_label: Dict[str, str]
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for canonical, bbox in character_entries:
        placeholder = f"${canonical}$"
        label = placeholder_to_label.get(placeholder, f"#{canonical}")
        payload.append(
            {
                "canonical": canonical,
                "label": label,
                "bbox": bbox,
            }
        )
    return payload


def format_character_pos_info(
    character_entries: List[Tuple[str, List[int]]], placeholder_to_label: Dict[str, str]
) -> Optional[str]:
    if not character_entries:
        return None
    parts: List[str] = []
    for canonical, bbox in character_entries:
        placeholder = f"${canonical}$"
        label = placeholder_to_label.get(placeholder, f"#{canonical}")
        x1, x2, y1, y2 = bbox
        parts.append(f"{label}:[{x1},{x2},{y1},{y2}]")
    if not parts:
        return None
    return "Characters: " + ", ".join(parts)


def build_glm_variants(
    glm_json: Optional[dict],
    artist_name: str,
    finaltag_dan_no_artist: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not glm_json or not isinstance(glm_json.get("image"), dict):
        return None, None, None

    character_entries, placeholder_to_label = parse_glm_characters(glm_json)
    characters_line = build_characters_line(character_entries)

    image_section = glm_json.get("image", {})
    caption_raw = str(image_section.get("caption", "")).strip()
    caption_repl = apply_placeholder_replacements(caption_raw, placeholder_to_label)
    caption_repl = caption_repl.strip()

    header = (
        "You are an assistant designed to generate anime images based on textual prompts. "
        "Your task is also to ensure that the characters in the image are positioned according to the normalized bounding box coordinates x1, x2, y1, y2 with values ranging from 1 to 1000. <Prompt Start> "
    )

    lines = [header]
    if characters_line:
        lines.append(characters_line)
    lines.append(f"Drawn by @{{{artist_name}}}")
    lines.append(caption_repl)

    v3 = "\\n".join(lines)

    # Variant 6: append danbooru after caption (within same single-line annotation using \n)
    v6 = v3 + "\\n" + finaltag_dan_no_artist

    # Variant 5 (XML) constructed elsewhere with tags; return characters for reuse
    return v3, v6, characters_line


def build_internlm_variant(
    internlm_json: Optional[dict],
    artist_name: str,
    characters_line_from_glm: Optional[str],
    glm_available: bool,
) -> Optional[str]:
    if not internlm_json or not isinstance(internlm_json.get("image"), dict):
        return None
    caption_raw = str(internlm_json.get("image", {}).get("caption", "")).strip()

    if glm_available and characters_line_from_glm:
        header = (
            "You are an assistant designed to generate anime images based on textual prompts. "
            "Your task is also to ensure that the characters in the image are positioned according to the normalized bounding box coordinates x1, x2, y1, y2 with values ranging from 1 to 1000. <Prompt Start> "
        )
        parts = [header, characters_line_from_glm, f"Drawn by @{{{artist_name}}}", caption_raw]
        return "\\n".join(parts)
    else:
        # GLM missing: simplified header and NO characters
        parts = [
            "You are an assistant designed to generate anime images based on textual prompts.",
            f"Drawn by @{{{artist_name}}}",
            caption_raw,
        ]
        return "\\n".join(parts)


def build_xml_variant(
    glm_json: Optional[dict],
    artist_name: str,
    characters_line_from_glm: Optional[str],
) -> Optional[str]:
    if not glm_json or not isinstance(glm_json.get("image"), dict):
        return None
    tags_raw = str(glm_json.get("image", {}).get("tags", "")).strip()

    # Apply placeholder replacements in tags as well
    character_entries, placeholder_to_label = parse_glm_characters(glm_json)
    tags_repl = apply_placeholder_replacements(tags_raw, placeholder_to_label)

    header = (
        "You are an assistant designed to generate anime images based on structured XML descriptions. "
        "Your task is also to ensure that the characters in the image are positioned according to the normalized bounding box coordinates x1, x2, y1, y2 with values ranging from 1 to 1000. <Prompt Start> "
    )

    parts = [header]
    if characters_line_from_glm:
        parts.append(characters_line_from_glm)
    parts.append(f"Drawn by @{{{artist_name}}}")
    parts.append(tags_repl)
    return "\\n".join(parts)


# ----------------------------
# Quality label aggregation across root
# ----------------------------
def collect_quality_labels(main_root: Path) -> Dict[str, Dict[str, str]]:
    all_image_scores: List[Tuple[str, str, float]] = []  # (artist, filename, score)

    for artist_dir in list_artist_dirs(main_root):
        results_path = artist_dir / "results.json"
        results = safe_read_json(results_path)
        if not results:
            continue
        for fname, data in results.items():
            try:
                score = float(data.get("final_score"))
            except Exception:
                continue
            all_image_scores.append((artist_dir.name, fname, score))

    all_image_scores.sort(key=lambda x: x[2], reverse=True)
    total = len(all_image_scores)
    if total == 0:
        return {}

    masterpiece_threshold = int(total * 0.15)
    best_quality_threshold = int(total * 0.40)
    bad_quality_threshold = int(total * 0.90)
    worst_quality_threshold = int(total * 0.95)

    quality_labels: Dict[str, Dict[str, str]] = {}
    for idx, (artist, fname, _) in enumerate(all_image_scores):
        quality_labels.setdefault(artist, {})
        if idx < masterpiece_threshold:
            quality_labels[artist][fname] = "masterpiece"
        elif idx < best_quality_threshold:
            quality_labels[artist][fname] = "best quality"
        elif bad_quality_threshold <= idx < worst_quality_threshold:
            quality_labels[artist][fname] = "bad quality"
        elif idx >= worst_quality_threshold:
            quality_labels[artist][fname] = "worst quality"
        else:
            quality_labels[artist][fname] = ""

    return quality_labels


# ----------------------------
# Image resizing & saving
# ----------------------------
def ensure_min_side(img: Image.Image, target_min_side: int) -> Image.Image:
    width, height = img.size
    min_side = min(width, height)
    if min_side == 0:
        return img
    if min_side == target_min_side:
        return img
    scale = target_min_side / float(min_side)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def save_as_webp(src_path: Path, dst_path: Path, target_min_side: int) -> bool:
    try:
        with Image.open(src_path) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            im = ensure_min_side(im, target_min_side)
            ensure_dir(dst_path.parent)
            im.save(str(dst_path), format="WEBP", quality=85, method=6)
        return True
    except Exception:
        return False


# ----------------------------
# Main processing per artist
# ----------------------------
def find_image_files_in_years(artist_dir: Path) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    for year_name in YEAR_FOLDERS:
        folder = artist_dir / year_name
        if not folder.exists() or not folder.is_dir():
            continue
        for entry in folder.iterdir():
            if entry.is_file():
                base, ext = splitext_lower(entry.name)
                if ext in VALID_IMAGE_EXTS:
                    results.append((year_name, entry))
        if year_name == "new":
            aug = folder / "Augmentation"
            if aug.exists() and aug.is_dir():
                for entry in aug.iterdir():
                    if entry.is_file():
                        base, ext = splitext_lower(entry.name)
                        if ext in VALID_IMAGE_EXTS:
                            # Keep relative path including Augmentation/ for filename key resolution
                            rel = Path("Augmentation") / entry.name
                            results.append((year_name, folder / rel))
    return results


def choose_unique_output_name(
    dst_artist_dir: Path,
    base_name: str,
    metadata_only: bool = False,
    original_extension: Optional[str] = None,
    used_names: Optional[Set[str]] = None,
) -> str:
    if metadata_only:
        extension = original_extension if original_extension else ""
    else:
        extension = ".webp"

    def build_candidate(counter: int) -> str:
        suffix = "" if counter == 1 else f"_{counter}"
        return f"{base_name}{suffix}{extension}"

    counter = 1
    while True:
        candidate = build_candidate(counter)
        if metadata_only:
            if used_names is None or candidate not in used_names:
                if used_names is not None:
                    used_names.add(candidate)
                return candidate
        else:
            if not (dst_artist_dir / candidate).exists():
                if used_names is not None:
                    used_names.add(candidate)
                return candidate
        counter += 1


def build_image_payload(
    nl_root: Path,
    artist_dir: Path,
    dst_artist_dir: Path,
    year_name: str,
    img_path: Path,
    image_index: int,
    assigned_img_name: str,
    results_json: Optional[dict],
    quality_labels_for_artist: Optional[Dict[str, str]],
    features_threshold: float,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    artist_display_name = artist_dir.name.replace("_", " ")
    final_artist_tag = fmt2danbooru(artist_dir.name)
    base_name = img_path.name
    stem = img_path.stem

    filename_for_analysis = (
        os.path.join("Augmentation", base_name) if "Augmentation" in img_path.parts else base_name
    )

    components = analyze_image_for_danbooru(
        artist_folder=artist_dir,
        year_folder_name=year_name,
        filename=filename_for_analysis,
        results_json=results_json,
        quality_labels_for_artist=quality_labels_for_artist,
        features_threshold=features_threshold,
    )
    danbooru_meta = compose_danbooru_metadata(components, final_artist_tag)

    results_entry = results_json.get(base_name) if results_json and base_name in results_json else {}
    finaltag_native = None
    if isinstance(results_entry, dict):
        for candidate_key in ("finaltag_native", "native_caption", "caption_native", "comment_native", "comment"):
            candidate_value = strip_or_none(results_entry.get(candidate_key))
            if candidate_value:
                finaltag_native = candidate_value
                break

    glm_json_path = nl_root / artist_dir.name / "NL_caption_GLM" / f"{stem}.json"
    internlm_json_path = nl_root / artist_dir.name / "NLcaption" / f"{stem}.json"
    glm_json = safe_read_json(glm_json_path)
    internlm_json = safe_read_json(internlm_json_path)

    character_entries_glm, placeholder_glm = parse_glm_characters(glm_json)
    character_entries_intern, placeholder_intern = parse_glm_characters(internlm_json)

    if character_entries_glm:
        character_entries = character_entries_glm
        placeholder_for_positions = placeholder_glm
    else:
        character_entries = character_entries_intern
        placeholder_for_positions = placeholder_intern

    character_pos_info = format_character_pos_info(character_entries, placeholder_for_positions)
    glm_characters_payload = build_characters_payload(character_entries, placeholder_for_positions)

    glm_nl_raw = extract_model_field(glm_json, "caption")
    glm_xml_raw = extract_model_field(glm_json, "tags")
    internlm_nl_raw = extract_model_field(internlm_json, "caption")
    internlm_xml_raw = extract_model_field(internlm_json, "tags")

    glm_placeholder_map = placeholder_glm
    intern_placeholder_map = placeholder_intern or placeholder_glm

    glm_nl_prompt = (
        collapse_whitespace(apply_placeholder_replacements(glm_nl_raw, glm_placeholder_map))
        if glm_nl_raw
        else None
    )
    glm_xml_prompt = (
        collapse_whitespace(apply_placeholder_replacements(glm_xml_raw, glm_placeholder_map))
        if glm_xml_raw
        else None
    )
    internlm_nl_prompt = (
        collapse_whitespace(apply_placeholder_replacements(internlm_nl_raw, intern_placeholder_map))
        if internlm_nl_raw
        else None
    )
    internlm_xml_prompt = (
        collapse_whitespace(apply_placeholder_replacements(internlm_xml_raw, intern_placeholder_map))
        if internlm_xml_raw
        else None
    )

    drawn_by_segment = f"Drawn by @{final_artist_tag}" if final_artist_tag else None
    artist_xml_segment = f"<artist_name>{final_artist_tag or ''}</artist_name>"

    prompts: Dict[str, Optional[str]] = {}
    prompts["GLM_NL_WITHPOS"] = (
        compose_prompt(
            GLM_NL_WITHPOS_HEADER,
            [drawn_by_segment, character_pos_info, glm_nl_prompt],
        )
        if character_pos_info and glm_nl_prompt
        else None
    )
    prompts["GLM_NL_WITHOUTPOS"] = (
        compose_prompt(GLM_NL_WITHOUTPOS_HEADER, [drawn_by_segment, glm_nl_prompt]) if glm_nl_prompt else None
    )
    prompts["GLM_XML"] = (
        compose_prompt(GLM_XML_HEADER, [artist_xml_segment, glm_xml_prompt]) if glm_xml_prompt else None
    )
    prompts["InternLM_NL_WITHPOS"] = (
        compose_prompt(
            INTERNLM_NL_WITHPOS_HEADER,
            [drawn_by_segment, character_pos_info, internlm_nl_prompt],
        )
        if character_pos_info and internlm_nl_prompt
        else None
    )
    prompts["InternLM_NL_WITHOUTPOS"] = (
        compose_prompt(INTERNLM_NL_WITHOUTPOS_HEADER, [drawn_by_segment, internlm_nl_prompt])
        if internlm_nl_prompt
        else None
    )
    prompts["InternLM_XML"] = (
        compose_prompt(INTERNLM_XML_HEADER, [artist_xml_segment, internlm_xml_prompt]) if internlm_xml_prompt else None
    )
    danbooru_prompt_body = strip_or_none(danbooru_meta.get("prompt_body"))
    prompts["Danbooru_tags"] = (
        compose_prompt(DANBOORU_HEADER, [danbooru_prompt_body]) if danbooru_prompt_body else None
    )

    available_prompt_types = [key for key in PROMPT_KEYS if prompts.get(key)]

    record: Dict[str, Any] = {
        "image_name": assigned_img_name,
        "source_image_name": base_name,
        "source_year": year_name,
        "image_index": image_index,
        "artist_display_name": artist_display_name,
        "final_artist_tag": strip_or_none(final_artist_tag),
        "quality_label": strip_or_none((quality_labels_for_artist or {}).get(base_name)),
        "glm_nl": glm_nl_raw,
        "internlm_nl": internlm_nl_raw,
        "glm_xml": glm_xml_raw,
        "internlm_xml": internlm_xml_raw,
        "glm_characters": glm_characters_payload,
        "character_pos_info": character_pos_info,
        "finaltag_dan": strip_or_none(danbooru_meta.get("finaltag_dan")),
        "finaltag_native": finaltag_native,
        "final_features_tag_prefix": strip_or_none(components.final_features_tag_prefix),
        "final_features_tag": strip_or_none(components.final_features_tag),
        "danbooru_prefix": strip_or_none(danbooru_meta.get("prefix")),
        "danbooru_suffix": strip_or_none(danbooru_meta.get("suffix")),
        "final_character_tag": strip_or_none(components.final_character_tag),
        "final_copyright_tag": strip_or_none(components.final_copyright_tag),
        "year_tag_specific": strip_or_none(components.year_tag_specific),
        "year_tag": strip_or_none(components.year_tag),
        "final_rating_tag": strip_or_none(components.final_rating_tag),
        "additional_tags": strip_or_none(components.additional_tags),
        "aes_rating": strip_or_none(components.aes_rating),
        "danbooru_prompt_body": danbooru_prompt_body,
        "prompts": prompts,
        "available_prompts": available_prompt_types,
    }

    return record, assigned_img_name, None


def process_artist(
    main_root: Path,
    nl_root: Path,
    artist_dir: Path,
    output_root: Path,
    quality_labels: Dict[str, Dict[str, str]],
    features_threshold: float,
    min_side: int,
    max_workers: int,
    metadata_only: bool,
    pbar: Optional[Any] = None,
) -> Tuple[int, int]:
    results_path = artist_dir / "results.json"
    results_json = safe_read_json(results_path)
    dst_artist_dir = output_root / artist_dir.name
    ensure_dir(dst_artist_dir)

    images = find_image_files_in_years(artist_dir)
    if not images:
        return 0, 0

    ordered_images = order_artist_images(images, results_json)
    indexed_images = [(idx, year, path) for idx, (year, path) in enumerate(ordered_images, start=1)]

    total = len(ordered_images)
    success = 0
    image_records: List[Dict[str, Any]] = []

    used_names: Set[str] = set()

    def handle_one(index_value: int, year_name: str, file_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        try:
            stem = file_path.stem
            assigned_name = choose_unique_output_name(
                dst_artist_dir=dst_artist_dir,
                base_name=stem,
                metadata_only=metadata_only,
                original_extension=file_path.suffix,
                used_names=used_names,
            )
            record, out_img_name, err = build_image_payload(
                nl_root=nl_root,
                artist_dir=artist_dir,
                dst_artist_dir=dst_artist_dir,
                year_name=year_name,
                img_path=file_path,
                image_index=index_value,
                assigned_img_name=assigned_name,
                results_json=results_json,
                quality_labels_for_artist=quality_labels.get(artist_dir.name, {}),
                features_threshold=features_threshold,
            )
            if err:
                return False, None, f"{file_path.name}: {err}"

            if not metadata_only:
                out_img_path = dst_artist_dir / out_img_name
                ok = save_as_webp(file_path, out_img_path, min_side)
                if not ok:
                    return False, None, f"{file_path.name}: failed to save webp"

            return True, record, out_img_name
        except Exception as e:
            return False, None, f"{file_path.name}: {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(handle_one, idx, year, path) for idx, year, path in indexed_images]
        for fut in as_completed(futures):
            ok, record, _msg = fut.result()
            if pbar is not None:
                pbar.update(1)
            if ok:
                success += 1
                if record:
                    image_records.append(record)

    image_records.sort(key=lambda rec: rec.get("image_index", 0))

    schema_version = 1
    artist_display_name = artist_dir.name.replace("_", " ")
    results_payload = {
        "schema_version": schema_version,
        "artist": artist_dir.name,
        "artist_display_name": artist_display_name,
        "generated_at": datetime.utcnow().isoformat(),
        "image_count": len(image_records),
        "prompt_types": PROMPT_KEYS,
        "images": {record["image_name"]: record for record in image_records},
    }

    output_json_path = dst_artist_dir / "results.json"
    output_json_path.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return success, total


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build training dataset with 6-style annotations and webp export")
    parser.add_argument("--main-root", type=str, required=True, help="Root path of the main dataset (artists with year folders and results.json)")
    parser.add_argument(
        "--nl-root",
        type=str,
        required=True,
        help="Root path of the natural language dataset (artists with NLcaption and NL_caption_GLM)",
    )
    parser.add_argument("--output-root", type=str, required=True, help="Destination root to write webp and txt preserving artist structure")
    parser.add_argument("--min-side", type=int, default=1600, help="Resize so that the minimum side equals this value")
    parser.add_argument("--features-threshold", type=float, default=FEATURES_THRESHOLD_DEFAULT, help="Threshold for features from results.json")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 8, help="Thread pool size")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only emit per-artist metadata JSON without writing converted images",
    )

    args = parser.parse_args(argv)
    main_root = Path(args.main_root)
    nl_root = Path(args.nl_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    # Aggregate quality labels
    quality_labels = collect_quality_labels(main_root)

    # Process each artist with a global progress bar
    artists = list_artist_dirs(main_root)
    total_success = 0
    total_images = sum(len(find_image_files_in_years(a)) for a in artists)
    with tqdm(total=total_images, desc="Total", unit="img") as pbar:
        for artist_dir in artists:
            ok_count, num = process_artist(
                main_root=main_root,
                nl_root=nl_root,
                artist_dir=artist_dir,
                output_root=output_root,
                quality_labels=quality_labels,
                features_threshold=args.features_threshold,
                min_side=args.min_side,
                max_workers=args.max_workers,
                metadata_only=args.metadata_only,
                pbar=pbar,
            )
            total_success += ok_count
            tqdm.write(f"Artist {artist_dir.name}: {ok_count}/{num} processed")

    tqdm.write(f"All done: {total_success}/{total_images} images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


