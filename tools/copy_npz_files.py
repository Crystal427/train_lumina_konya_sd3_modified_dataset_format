"""
Copy npz files from processed directory back to original output directory.

This script copies npz files (e.g., *_lumina.npz) from a processed directory
back to the original output directory, preserving the artist folder structure.
It matches npz files with their corresponding webp files by extracting the base name.
It also copies corresponding txt files if they exist.

Features:
- Copies npz files only if they don't already exist in destination
- Always copies txt files, overwriting existing ones
- Matches files by base name with webp files in destination

Example:
    Source: basename_1600x2607_lumina.npz, basename.txt
    Matches: basename.webp in destination
    Copies: basename_1600x2607_lumina.npz (if not exists), basename.txt (always)
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


def extract_base_name_from_npz(npz_filename: str) -> Optional[str]:
    """
    Extract base name from npz filename.
    
    Example:
        gelbooru_8904448_47d23a5c3cd05d1e094b4e465a0ed1e3_R73Pq6Cs_1600x2607_lumina.npz
        -> gelbooru_8904448_47d23a5c3cd05d1e094b4e465a0ed1e3_R73Pq6Cs
    
    Args:
        npz_filename: Name of the npz file
        
    Returns:
        Base name without resolution and suffix, or None if pattern doesn't match
    """
    # Pattern: basename_WIDTHxHEIGHT_lumina.npz
    pattern = r'^(.+?)_\d+x\d+_lumina\.npz$'
    match = re.match(pattern, npz_filename)
    
    if match:
        return match.group(1)
    
    return None


def build_webp_index(dest_root: Path) -> Dict[str, Path]:
    """
    Build an index of all webp files in destination directory.
    
    Args:
        dest_root: Destination root directory
        
    Returns:
        Dictionary mapping (artist_name, base_name) -> webp_file_path
    """
    webp_index: Dict[str, Path] = {}
    
    if not dest_root.exists():
        return webp_index
    
    for artist_dir in dest_root.iterdir():
        if not artist_dir.is_dir() or artist_dir.name.startswith('.'):
            continue
        
        for webp_file in artist_dir.glob("*.webp"):
            if webp_file.is_file():
                base_name = webp_file.stem  # filename without extension
                key = f"{artist_dir.name}/{base_name}"
                webp_index[key] = webp_file
    
    return webp_index


def find_npz_files(root: Path, pattern: str = "*_lumina.npz") -> List[Tuple[Path, Path, str, Optional[Path]]]:
    """
    Find all npz files matching the pattern in root directory.
    
    Args:
        root: Root directory to search
        pattern: Glob pattern for npz files
        
    Returns:
        List of (artist_dir, npz_file_path, base_name, txt_file_path) tuples
    """
    results: List[Tuple[Path, Path, str, Optional[Path]]] = []
    
    if not root.exists():
        print(f"Warning: Source directory does not exist: {root}")
        return results
    
    # Iterate through artist directories
    for artist_dir in root.iterdir():
        if not artist_dir.is_dir():
            continue
        
        # Skip special directories
        if artist_dir.name.startswith('.'):
            continue
            
        # Find all npz files in this artist directory
        for npz_file in artist_dir.glob(pattern):
            if npz_file.is_file():
                base_name = extract_base_name_from_npz(npz_file.name)
                if base_name:
                    # Check if corresponding txt file exists
                    txt_file = artist_dir / f"{base_name}.txt"
                    txt_file_path = txt_file if txt_file.exists() else None
                    results.append((artist_dir, npz_file, base_name, txt_file_path))
    
    return results


def copy_npz_with_structure(
    source_root: Path,
    dest_root: Path,
    pattern: str = "*_lumina.npz",
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int, int, int]:
    """
    Copy npz files from source to destination, only if matching webp exists.
    Also copy corresponding txt files if they exist.
    
    Args:
        source_root: Source directory containing npz files
        dest_root: Destination directory to copy files to
        pattern: Glob pattern for npz files
        dry_run: If True, only show what would be copied without actually copying
        verbose: If True, show detailed matching information
        
    Returns:
        Tuple of (npz_copied, npz_skipped, total_npz, txt_copied, txt_skipped)
    """
    print("Building index of webp files in destination directory...")
    webp_index = build_webp_index(dest_root)
    print(f"Found {len(webp_index)} webp files in destination")
    
    print("\nScanning for npz files in source directory...")
    npz_files = find_npz_files(source_root, pattern)
    
    if not npz_files:
        print(f"No npz files matching pattern '{pattern}' found in {source_root}")
        return 0, 0, 0, 0, 0
    
    npz_copied = 0
    npz_skipped = 0
    txt_copied = 0
    txt_skipped = 0
    total_count = len(npz_files)
    
    print(f"Found {total_count} npz files")
    if dry_run:
        print("DRY RUN MODE - No files will be copied\n")
    else:
        print()
    
    for artist_dir, npz_file, base_name, txt_file in tqdm(npz_files, desc="Processing files", unit="file"):
        artist_name = artist_dir.name
        
        # Build the key to look up in webp_index
        key = f"{artist_name}/{base_name}"
        
        # Check if corresponding webp exists in destination
        if key not in webp_index:
            if verbose:
                tqdm.write(f"Skip (no match): {npz_file.name} (looking for {base_name}.webp)")
            npz_skipped += 1
            if txt_file:
                txt_skipped += 1
            continue
        
        # Destination paths
        dest_artist_dir = dest_root / artist_name
        dest_npz_path = dest_artist_dir / npz_file.name
        
        # Ensure destination directory exists
        if not dest_artist_dir.exists():
            if not dry_run:
                dest_artist_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if npz file already exists in destination
        npz_already_exists = dest_npz_path.exists()
        
        # Copy npz file only if it doesn't exist in destination
        if npz_already_exists:
            if verbose:
                tqdm.write(f"Skip npz (exists): {npz_file.name}")
            npz_skipped += 1
        else:
            try:
                if dry_run:
                    tqdm.write(f"Would copy npz: {npz_file.name}")
                    if verbose:
                        tqdm.write(f"  From: {npz_file}")
                        tqdm.write(f"  To:   {dest_npz_path}")
                else:
                    # Copy the npz file
                    shutil.copy2(npz_file, dest_npz_path)
                    if verbose:
                        tqdm.write(f"Copied npz: {npz_file.name} -> {dest_npz_path}")
                
                npz_copied += 1
                
            except Exception as e:
                tqdm.write(f"Error copying npz {npz_file.name}: {e}")
                npz_skipped += 1
        
        # Always copy txt file if it exists (overwrite)
        if txt_file:
            dest_txt_path = dest_artist_dir / txt_file.name
            try:
                if dry_run:
                    tqdm.write(f"Would copy txt: {txt_file.name}")
                    if verbose:
                        tqdm.write(f"  From: {txt_file}")
                        tqdm.write(f"  To:   {dest_txt_path}")
                else:
                    # Copy the txt file (overwrite if exists)
                    shutil.copy2(txt_file, dest_txt_path)
                    if verbose:
                        tqdm.write(f"Copied txt: {txt_file.name} -> {dest_txt_path}")
                
                txt_copied += 1
                
            except Exception as e:
                tqdm.write(f"Error copying txt {txt_file.name}: {e}")
                txt_skipped += 1
    
    return npz_copied, npz_skipped, total_count, txt_copied, txt_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Copy npz files from processed directory back to original output directory, "
        "matching them with existing webp files. Also copies corresponding txt files. "
        "NPZ files are skipped if they already exist in destination, but txt files are always copied."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source directory containing npz files (processed directory)",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        required=True,
        help="Destination directory to copy npz files to (original output directory)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_lumina.npz",
        help="Glob pattern for npz files (default: *_lumina.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed matching and copying information",
    )
    
    args = parser.parse_args()
    
    source_root = Path(args.source_dir)
    dest_root = Path(args.dest_dir)
    
    if not source_root.exists():
        print(f"Error: Source directory does not exist: {source_root}")
        return 1
    
    if not dest_root.exists():
        print(f"Error: Destination directory does not exist: {dest_root}")
        return 1
    
    print(f"Source directory: {source_root}")
    print(f"Destination directory: {dest_root}")
    print(f"Pattern: {args.pattern}")
    print()
    
    npz_copied, npz_skipped, total_npz, txt_copied, txt_skipped = copy_npz_with_structure(
        source_root=source_root,
        dest_root=dest_root,
        pattern=args.pattern,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    print()
    print("=" * 60)
    print("NPZ Files:")
    print(f"  Total found: {total_npz}")
    print(f"  Copied: {npz_copied}")
    print(f"  Skipped: {npz_skipped}")
    
    if npz_copied > 0:
        copy_rate = (npz_copied / total_npz) * 100
        print(f"  Copy rate: {copy_rate:.1f}%")
    
    print("\nTXT Files:")
    print(f"  Copied: {txt_copied}")
    if txt_skipped > 0:
        print(f"  Skipped: {txt_skipped}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually copy files.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

