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
def splitext_lower(name: str) -> tuple[str, str]:
    """Split filename and return lowercase extension."""
    base, ext = os.path.splitext(name)
    return base, ext.lower()


def load_whitelist(json_path: Path) -> Set[str]:
    """Load image stems whitelist from JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support both list and dict format
        if isinstance(data, list):
            return set(data)
        elif isinstance(data, dict) and "image_stems" in data:
            return set(data["image_stems"])
        else:
            print(f"Error: Invalid JSON format in {json_path}", file=sys.stderr)
            return set()
    except Exception as e:
        print(f"Error loading whitelist: {e}", file=sys.stderr)
        return set()


def find_related_files(image_path: Path) -> List[Path]:
    """
    Find related files for an image:
    - Same stem with .txt extension
    - Same stem with pattern: stem_*suffix.npz (e.g., stem_1988x1600_lumina.npz)
    """
    related = []
    stem = image_path.stem
    parent = image_path.parent
    
    # Find .txt file
    txt_file = parent / f"{stem}.txt"
    if txt_file.exists():
        related.append(txt_file)
    
    # Find .npz files with pattern: stem_*.npz
    try:
        for file in parent.iterdir():
            if file.is_file() and file.suffix.lower() == ".npz":
                # Check if filename starts with stem followed by underscore
                if file.stem.startswith(f"{stem}_"):
                    related.append(file)
    except Exception:
        pass
    
    return related


def delete_file_safe(file_path: Path) -> bool:
    """Delete file safely."""
    try:
        file_path.unlink()
        return True
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}", file=sys.stderr)
        return False


def process_directory(
    dataset_dir: Path,
    whitelist: Set[str],
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Process dataset directory and delete images not in whitelist.
    Returns: (deleted_images, deleted_related_files, kept_images)
    """
    deleted_images = 0
    deleted_related = 0
    kept_images = 0
    
    # Collect all image files
    all_images: List[Path] = []
    
    print("Scanning for image files...")
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            base, ext = splitext_lower(filename)
            if ext in VALID_IMAGE_EXTS:
                all_images.append(Path(root) / filename)
    
    print(f"Found {len(all_images)} image files")
    
    with tqdm(total=len(all_images), desc="Processing", unit="img") as pbar:
        for img_path in all_images:
            stem = img_path.stem
            
            if stem in whitelist:
                # Keep this image
                kept_images += 1
                pbar.set_postfix(kept=kept_images, deleted=deleted_images)
            else:
                # Delete this image and related files
                files_to_delete = [img_path] + find_related_files(img_path)
                
                if dry_run:
                    pbar.write(f"[DRY RUN] Would delete: {img_path.name} and {len(files_to_delete)-1} related files")
                    deleted_images += 1
                    deleted_related += len(files_to_delete) - 1
                else:
                    # Delete image
                    if delete_file_safe(img_path):
                        deleted_images += 1
                        pbar.write(f"Deleted: {img_path.name}")
                        
                        # Delete related files
                        for related_file in files_to_delete[1:]:
                            if delete_file_safe(related_file):
                                deleted_related += 1
                                pbar.write(f"  ├─ {related_file.name}")
            
            pbar.update(1)
    
    return deleted_images, deleted_related, kept_images


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean dataset by removing images not in whitelist (and their .txt and .npz files)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root path of the dataset to clean",
    )
    parser.add_argument(
        "--whitelist-json",
        type=str,
        required=True,
        help="Path to JSON file containing image stems whitelist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be deleted without actually deleting",
    )
    
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    whitelist_json = Path(args.whitelist_json)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1
    
    if not whitelist_json.exists():
        print(f"Error: Whitelist JSON does not exist: {whitelist_json}", file=sys.stderr)
        return 1
    
    # Load whitelist
    whitelist = load_whitelist(whitelist_json)
    if not whitelist:
        print("Error: Empty or invalid whitelist", file=sys.stderr)
        return 1
    
    print(f"Loaded whitelist with {len(whitelist)} image stems")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be deleted ***\n")
    
    # Process dataset
    deleted_imgs, deleted_related, kept = process_directory(
        dataset_dir,
        whitelist,
        dry_run=args.dry_run,
    )
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Kept images: {kept}")
    print(f"  Deleted images: {deleted_imgs}")
    print(f"  Deleted related files: {deleted_related}")
    print(f"  Total deleted: {deleted_imgs + deleted_related}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

