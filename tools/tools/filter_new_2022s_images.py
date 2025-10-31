import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set
from tqdm import tqdm


# ----------------------------
# Constants
# ----------------------------
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TARGET_YEAR_FOLDERS = ["new", "2022s"]


# ----------------------------
# Utility helpers
# ----------------------------
def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def list_artist_dirs(root: Path) -> List[Path]:
    """List all artist directories in the root."""
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def splitext_lower(name: str) -> tuple[str, str]:
    """Split filename and return lowercase extension."""
    base, ext = os.path.splitext(name)
    return base, ext.lower()


def copy_file_safe(src: Path, dst: Path) -> bool:
    """Copy file safely, create parent directory if needed."""
    try:
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Failed to copy {src} to {dst}: {e}", file=sys.stderr)
        return False


def find_images_in_target_years(artist_dir: Path) -> List[tuple[str, Path]]:
    """
    Find all images in 'new' and '2022s' folders (including Augmentation).
    If 'new' folder has more than 200 images, only return images from 'new'.
    Returns: List of (year_name, image_path) tuples.
    """
    # First, count images in 'new' folder
    new_images: List[tuple[str, Path]] = []
    new_folder = artist_dir / "new"
    
    if new_folder.exists() and new_folder.is_dir():
        # Images directly in new folder
        for entry in new_folder.iterdir():
            if entry.is_file():
                base, ext = splitext_lower(entry.name)
                if ext in VALID_IMAGE_EXTS:
                    new_images.append(("new", entry))
        
        # Images in Augmentation subfolder
        aug = new_folder / "Augmentation"
        if aug.exists() and aug.is_dir():
            for entry in aug.iterdir():
                if entry.is_file():
                    base, ext = splitext_lower(entry.name)
                    if ext in VALID_IMAGE_EXTS:
                        new_images.append(("new", entry))
    
    # If new folder has more than 200 images, only return new images
    if len(new_images) > 200:
        return new_images
    
    # Otherwise, collect images from both 'new' and '2022s'
    results: List[tuple[str, Path]] = new_images.copy()
    
    # Add images from 2022s folder
    folder_2022s = artist_dir / "2022s"
    if folder_2022s.exists() and folder_2022s.is_dir():
        for entry in folder_2022s.iterdir():
            if entry.is_file():
                base, ext = splitext_lower(entry.name)
                if ext in VALID_IMAGE_EXTS:
                    results.append(("2022s", entry))
    
    return results


def find_matching_json(jsons_folder: Path, image_stem: str) -> Path | None:
    """
    Find matching JSON file for an image (match first 16 chars of stem).
    """
    if not jsons_folder.exists():
        return None
    
    try:
        name_prefix = image_stem[:16]
        for json_file in os.listdir(jsons_folder):
            if not json_file.endswith((".json", ".png.json", ".jpg.json", ".jpeg.json", ".webp.json")):
                continue
            if json_file.startswith(name_prefix):
                candidate = jsons_folder / json_file
                if candidate.exists():
                    return candidate
    except Exception:
        pass
    
    return None


def process_artist(
    artist_dir: Path,
    output_root: Path,
) -> tuple[int, int]:
    """
    Process one artist directory: copy new/2022s images, jsons, and results.json.
    Returns: (success_count, total_count)
    """
    images = find_images_in_target_years(artist_dir)
    if not images:
        return 0, 0
    
    dst_artist_dir = output_root / artist_dir.name
    ensure_dir(dst_artist_dir)
    
    # Copy results.json
    results_json_src = artist_dir / "results.json"
    if results_json_src.exists():
        results_json_dst = dst_artist_dir / "results.json"
        copy_file_safe(results_json_src, results_json_dst)
    
    # Prepare jsons folder
    jsons_src = artist_dir / "jsons"
    jsons_dst = dst_artist_dir / "jsons"
    
    success = 0
    copied_json_files: Set[str] = set()
    
    for year_name, img_path in images:
        # Determine destination path
        if "Augmentation" in img_path.parts:
            # Handle Augmentation subfolder
            rel_to_year = img_path.relative_to(artist_dir / year_name)
            dst_img_path = dst_artist_dir / year_name / rel_to_year
        else:
            dst_img_path = dst_artist_dir / year_name / img_path.name
        
        # Copy image
        if copy_file_safe(img_path, dst_img_path):
            success += 1
            
            # Copy matching JSON file (avoid duplicates)
            if jsons_src.exists():
                json_file = find_matching_json(jsons_src, img_path.stem)
                if json_file and json_file.name not in copied_json_files:
                    json_dst = jsons_dst / json_file.name
                    if copy_file_safe(json_file, json_dst):
                        copied_json_files.add(json_file.name)
    
    return success, len(images)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter and copy images from 'new' and '2022s' folders to a separate directory"
    )
    parser.add_argument(
        "--main-root",
        type=str,
        required=True,
        help="Root path of the main dataset (artists with year folders)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Destination root to copy filtered images",
    )
    
    args = parser.parse_args(argv)
    main_root = Path(args.main_root)
    output_root = Path(args.output_root)
    
    if not main_root.exists():
        print(f"Error: Main root directory does not exist: {main_root}", file=sys.stderr)
        return 1
    
    ensure_dir(output_root)
    
    # Process all artists
    artists = list_artist_dirs(main_root)
    total_success = 0
    total_images = 0
    
    print(f"Found {len(artists)} artist directories")
    print(f"Filtering images from 'new' and '2022s' folders...")
    
    with tqdm(total=len(artists), desc="Processing artists", unit="artist") as pbar:
        for artist_dir in artists:
            ok_count, num = process_artist(artist_dir, output_root)
            total_success += ok_count
            total_images += num
            pbar.update(1)
            if num > 0:
                pbar.write(f"Artist {artist_dir.name}: {ok_count}/{num} images copied")
    
    print(f"\nAll done: {total_success}/{total_images} images copied")
    print(f"Output directory: {output_root}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

