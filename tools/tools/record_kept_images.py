import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Set
from tqdm import tqdm


# ----------------------------
# Constants
# ----------------------------
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ----------------------------
# Utility helpers
# ----------------------------
def list_artist_dirs(root: Path) -> List[Path]:
    """List all artist directories in the root."""
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def splitext_lower(name: str) -> tuple[str, str]:
    """Split filename and return lowercase extension."""
    base, ext = os.path.splitext(name)
    return base, ext.lower()


def collect_image_stems(artist_dir: Path) -> Set[str]:
    """
    Collect all image file stems (filename without extension) from an artist directory.
    Includes images in 'new', '2022s' folders and their Augmentation subfolders.
    """
    stems: Set[str] = set()
    
    # Check all subdirectories in artist folder
    for subdir in artist_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Process images directly in the folder
        for entry in subdir.iterdir():
            if entry.is_file():
                base, ext = splitext_lower(entry.name)
                if ext in VALID_IMAGE_EXTS:
                    stems.add(base)
        
        # Check for Augmentation subfolder
        aug = subdir / "Augmentation"
        if aug.exists() and aug.is_dir():
            for entry in aug.iterdir():
                if entry.is_file():
                    base, ext = splitext_lower(entry.name)
                    if ext in VALID_IMAGE_EXTS:
                        stems.add(base)
    
    return stems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record all image filenames (stems) from filtered dataset into a JSON whitelist"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root path of the filtered dataset (artists with new/2022s images)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to output JSON file containing image stems whitelist",
    )
    
    args = parser.parse_args(argv)
    input_root = Path(args.input_root)
    output_json = Path(args.output_json)
    
    if not input_root.exists():
        print(f"Error: Input root directory does not exist: {input_root}", file=sys.stderr)
        return 1
    
    # Collect all image stems across all artists
    all_stems: Set[str] = set()
    artists = list_artist_dirs(input_root)
    
    print(f"Found {len(artists)} artist directories")
    print(f"Collecting image filenames...")
    
    artist_stats = {}
    
    with tqdm(total=len(artists), desc="Scanning artists", unit="artist") as pbar:
        for artist_dir in artists:
            stems = collect_image_stems(artist_dir)
            if stems:
                all_stems.update(stems)
                artist_stats[artist_dir.name] = len(stems)
                pbar.write(f"Artist {artist_dir.name}: {len(stems)} images")
            pbar.update(1)
    
    # Save to JSON
    output_data = {
        "total_images": len(all_stems),
        "total_artists": len(artist_stats),
        "artist_stats": artist_stats,
        "image_stems": sorted(list(all_stems)),
    }
    
    # Ensure parent directory exists
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll done: {len(all_stems)} unique image stems recorded")
    print(f"Output JSON: {output_json}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

