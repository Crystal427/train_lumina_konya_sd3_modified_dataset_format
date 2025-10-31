"""
Copy npz files from processed directory back to original output directory.

This script copies npz files (e.g., *_lumina.npz) from a processed directory
back to the original output directory, preserving the artist folder structure.
It matches npz files with their corresponding webp files by extracting the base name.

Example:
    Source: basename_1600x2607_lumina.npz
    Matches: basename.webp in destination
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


def find_npz_files(root: Path, pattern: str = "*_lumina.npz") -> List[Tuple[Path, Path, str]]:
    """
    Find all npz files matching the pattern in root directory.
    
    Args:
        root: Root directory to search
        pattern: Glob pattern for npz files
        
    Returns:
        List of (artist_dir, npz_file_path, base_name) tuples
    """
    results: List[Tuple[Path, Path, str]] = []
    
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
                    results.append((artist_dir, npz_file, base_name))
    
    return results


def copy_npz_with_structure(
    source_root: Path,
    dest_root: Path,
    pattern: str = "*_lumina.npz",
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Copy npz files from source to destination, only if matching webp exists.
    
    Args:
        source_root: Source directory containing npz files
        dest_root: Destination directory to copy files to
        pattern: Glob pattern for npz files
        dry_run: If True, only show what would be copied without actually copying
        verbose: If True, show detailed matching information
        
    Returns:
        Tuple of (success_count, skipped_count, total_count)
    """
    print("Building index of webp files in destination directory...")
    webp_index = build_webp_index(dest_root)
    print(f"Found {len(webp_index)} webp files in destination")
    
    print("\nScanning for npz files in source directory...")
    npz_files = find_npz_files(source_root, pattern)
    
    if not npz_files:
        print(f"No npz files matching pattern '{pattern}' found in {source_root}")
        return 0, 0, 0
    
    success_count = 0
    skipped_count = 0
    total_count = len(npz_files)
    
    print(f"Found {total_count} npz files")
    if dry_run:
        print("DRY RUN MODE - No files will be copied\n")
    else:
        print()
    
    for artist_dir, npz_file, base_name in tqdm(npz_files, desc="Processing npz files", unit="file"):
        artist_name = artist_dir.name
        
        # Build the key to look up in webp_index
        key = f"{artist_name}/{base_name}"
        
        # Check if corresponding webp exists in destination
        if key not in webp_index:
            if verbose:
                tqdm.write(f"Skip (no match): {npz_file.name} (looking for {base_name}.webp)")
            skipped_count += 1
            continue
        
        # Destination paths
        dest_artist_dir = dest_root / artist_name
        dest_npz_path = dest_artist_dir / npz_file.name
        
        # Ensure destination directory exists
        if not dest_artist_dir.exists():
            if not dry_run:
                dest_artist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if dry_run:
                tqdm.write(f"Would copy: {npz_file.name}")
                if verbose:
                    tqdm.write(f"  Matched with: {base_name}.webp")
                    tqdm.write(f"  From: {npz_file}")
                    tqdm.write(f"  To:   {dest_npz_path}")
            else:
                # Copy the file
                shutil.copy2(npz_file, dest_npz_path)
                if verbose:
                    tqdm.write(f"Copied: {npz_file.name} -> {dest_npz_path}")
            
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"Error copying {npz_file.name}: {e}")
            skipped_count += 1
            continue
    
    return success_count, skipped_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Copy npz files from processed directory back to original output directory, "
        "matching them with existing webp files"
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
    
    success_count, skipped_count, total_count = copy_npz_with_structure(
        source_root=source_root,
        dest_root=dest_root,
        pattern=args.pattern,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    print()
    print("=" * 60)
    print(f"Total npz files found: {total_count}")
    print(f"Files copied: {success_count}")
    print(f"Files skipped (no matching webp): {skipped_count}")
    
    if success_count > 0:
        match_rate = (success_count / total_count) * 100
        print(f"Match rate: {match_rate:.1f}%")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually copy files.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

