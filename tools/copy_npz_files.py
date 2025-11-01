"""
Copy npz files from processed directory back to original output directory.

This script copies npz files (e.g., *_lumina.npz) from a processed directory
back to the original output directory, preserving the artist folder structure.
It matches npz files with their corresponding webp files using a flexible matching rule:
- If filename >= 50 chars: match first 50 characters
- If filename < 50 chars: match everything before the last underscore

The script renames npz files to include the random number from the target webp file.

Features:
- Matches files by prefix (first 50 chars or before last underscore)
- Renames npz files to include target webp's random number
- Skips files with multiple matches (ambiguous)
- Always copies txt files, overwriting existing ones

Example:
    Source: basename_12345.webp, basename_1600x2607_lumina.npz, basename.txt
    Target: basename_67890.webp
    Result: basename_67890_1600x2607_lumina.npz (renamed), basename.txt (copied)
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from tqdm import tqdm


def extract_matching_key(filename: str) -> str:
    """
    Extract matching key from filename for pairing.
    
    Rules:
    - If filename >= 50 chars: return first 50 characters
    - If filename < 50 chars: return everything before the last underscore
    
    Args:
        filename: Name of the file (without extension)
        
    Returns:
        Matching key for pairing
    """
    if len(filename) >= 50:
        return filename[:50]
    else:
        # Find last underscore and return everything before it
        last_underscore_idx = filename.rfind('_')
        if last_underscore_idx > 0:
            return filename[:last_underscore_idx]
        else:
            # No underscore found, return the whole filename
            return filename


def extract_npz_suffix(npz_filename: str) -> Optional[str]:
    """
    Extract the suffix part from npz filename (e.g., _1600x2607_lumina.npz).
    
    Args:
        npz_filename: Name of the npz file
        
    Returns:
        Suffix part (including leading underscore) or None if pattern doesn't match
    """
    # Pattern: _WIDTHxHEIGHT_lumina.npz or _xxx_lumina.npz
    pattern = r'(_\d+x\d+_lumina\.npz)$'
    match = re.search(pattern, npz_filename)
    
    if match:
        return match.group(1)
    
    # Try alternative pattern: _xxx_lumina.npz
    pattern2 = r'(_.+?_lumina\.npz)$'
    match2 = re.search(pattern2, npz_filename)
    if match2:
        return match2.group(1)
    
    return None


def extract_webp_random_suffix(webp_filename: str) -> Optional[str]:
    """
    Extract the random number suffix from webp filename (e.g., _67890 from basename_67890.webp).
    
    Args:
        webp_filename: Name of the webp file (without extension)
        
    Returns:
        Random suffix (including leading underscore) or empty string if no suffix
    """
    # Extract everything after the matching key
    matching_key = extract_matching_key(webp_filename)
    if len(webp_filename) > len(matching_key):
        return webp_filename[len(matching_key):]
    return ""


def build_webp_index(dest_root: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Build an index of all webp files in destination directory.
    
    Args:
        dest_root: Destination root directory
        
    Returns:
        Dictionary mapping (artist_name, matching_key) -> [(webp_file_path, webp_stem), ...]
    """
    webp_index: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    
    if not dest_root.exists():
        return webp_index
    
    for artist_dir in dest_root.iterdir():
        if not artist_dir.is_dir() or artist_dir.name.startswith('.'):
            continue
        
        for webp_file in artist_dir.glob("*.webp"):
            if webp_file.is_file():
                webp_stem = webp_file.stem  # filename without extension
                matching_key = extract_matching_key(webp_stem)
                key = f"{artist_dir.name}/{matching_key}"
                webp_index[key].append((webp_file, webp_stem))
    
    return webp_index


def find_npz_files(root: Path, pattern: str = "*_lumina.npz") -> List[Tuple[Path, Path, str, str, Optional[Path]]]:
    """
    Find all npz files matching the pattern in root directory.
    
    Args:
        root: Root directory to search
        pattern: Glob pattern for npz files
        
    Returns:
        List of (artist_dir, npz_file_path, matching_key, npz_suffix, txt_file_path) tuples
    """
    results: List[Tuple[Path, Path, str, str, Optional[Path]]] = []
    
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
                npz_suffix = extract_npz_suffix(npz_file.name)
                if npz_suffix:
                    # Remove suffix to get the base part
                    base_part = npz_file.name[:-len(npz_suffix)]
                    matching_key = extract_matching_key(base_part)
                    
                    # Check if corresponding txt file exists
                    # txt file should match the base part (before suffix)
                    txt_file = artist_dir / f"{base_part}.txt"
                    txt_file_path = txt_file if txt_file.exists() else None
                    results.append((artist_dir, npz_file, matching_key, npz_suffix, txt_file_path))
    
    return results


def copy_npz_with_structure(
    source_root: Path,
    dest_root: Path,
    pattern: str = "*_lumina.npz",
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int, int, int, int]:
    """
    Copy npz files from source to destination with renamed filenames to match target webp.
    Also copy corresponding txt files if they exist.
    
    The function:
    1. Matches npz and webp files by their matching key (first 50 chars or before last underscore)
    2. Renames npz files to include the random suffix from target webp
    3. Skips files with multiple matches (ambiguous)
    
    Args:
        source_root: Source directory containing npz files
        dest_root: Destination directory to copy files to
        pattern: Glob pattern for npz files
        dry_run: If True, only show what would be copied without actually copying
        verbose: If True, show detailed matching information
        
    Returns:
        Tuple of (npz_copied, npz_skipped_no_match, npz_skipped_multi_match, total_npz, txt_copied, txt_skipped)
    """
    print("Building index of webp files in destination directory...")
    webp_index = build_webp_index(dest_root)
    
    # Count unique keys and total webp files
    unique_keys = len(webp_index)
    total_webp_files = sum(len(matches) for matches in webp_index.values())
    print(f"Found {total_webp_files} webp files with {unique_keys} unique matching keys in destination")
    
    print("\nScanning for npz files in source directory...")
    npz_files = find_npz_files(source_root, pattern)
    
    if not npz_files:
        print(f"No npz files matching pattern '{pattern}' found in {source_root}")
        return 0, 0, 0, 0, 0, 0
    
    npz_copied = 0
    npz_skipped_no_match = 0
    npz_skipped_multi_match = 0
    txt_copied = 0
    txt_skipped = 0
    total_count = len(npz_files)
    
    print(f"Found {total_count} npz files")
    if dry_run:
        print("DRY RUN MODE - No files will be copied\n")
    else:
        print()
    
    for artist_dir, npz_file, matching_key, npz_suffix, txt_file in tqdm(npz_files, desc="Processing files", unit="file"):
        artist_name = artist_dir.name
        
        # Build the key to look up in webp_index
        key = f"{artist_name}/{matching_key}"
        
        # Check if corresponding webp exists in destination
        if key not in webp_index:
            if verbose:
                tqdm.write(f"Skip (no match): {npz_file.name} (key: {matching_key})")
            npz_skipped_no_match += 1
            if txt_file:
                txt_skipped += 1
            continue
        
        # Get matching webp files
        matching_webps = webp_index[key]
        
        # Skip if multiple matches (ambiguous)
        if len(matching_webps) > 1:
            if verbose:
                tqdm.write(f"Skip (multiple matches): {npz_file.name} has {len(matching_webps)} matching webp files")
            npz_skipped_multi_match += 1
            if txt_file:
                txt_skipped += 1
            continue
        
        # Get the single matching webp
        webp_path, webp_stem = matching_webps[0]
        
        # Extract random suffix from target webp
        webp_random_suffix = extract_webp_random_suffix(webp_stem)
        
        # Construct new npz filename: matching_key + webp_random_suffix + npz_suffix
        new_npz_name = f"{matching_key}{webp_random_suffix}{npz_suffix}"
        
        # Destination paths
        dest_artist_dir = dest_root / artist_name
        dest_npz_path = dest_artist_dir / new_npz_name
        
        # Ensure destination directory exists
        if not dest_artist_dir.exists():
            if not dry_run:
                dest_artist_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if npz file already exists in destination
        npz_already_exists = dest_npz_path.exists()
        
        # Copy npz file only if it doesn't exist in destination
        if npz_already_exists:
            if verbose:
                tqdm.write(f"Skip npz (exists): {new_npz_name}")
            npz_skipped_no_match += 1
        else:
            try:
                if dry_run:
                    tqdm.write(f"Would copy npz: {npz_file.name} -> {new_npz_name}")
                    if verbose:
                        tqdm.write(f"  From: {npz_file}")
                        tqdm.write(f"  To:   {dest_npz_path}")
                else:
                    # Copy the npz file with new name
                    shutil.copy2(npz_file, dest_npz_path)
                    if verbose:
                        tqdm.write(f"Copied npz: {npz_file.name} -> {new_npz_name}")
                
                npz_copied += 1
                
            except Exception as e:
                tqdm.write(f"Error copying npz {npz_file.name}: {e}")
                npz_skipped_no_match += 1
        
        # Always copy txt file if it exists (overwrite)
        if txt_file:
            # Construct new txt filename based on new npz name (without _lumina.npz suffix)
            new_txt_base = new_npz_name.replace('_lumina.npz', '')
            new_txt_name = f"{new_txt_base}.txt"
            dest_txt_path = dest_artist_dir / new_txt_name
            
            try:
                if dry_run:
                    tqdm.write(f"Would copy txt: {txt_file.name} -> {new_txt_name}")
                    if verbose:
                        tqdm.write(f"  From: {txt_file}")
                        tqdm.write(f"  To:   {dest_txt_path}")
                else:
                    # Copy the txt file (overwrite if exists)
                    shutil.copy2(txt_file, dest_txt_path)
                    if verbose:
                        tqdm.write(f"Copied txt: {txt_file.name} -> {new_txt_name}")
                
                txt_copied += 1
                
            except Exception as e:
                tqdm.write(f"Error copying txt {txt_file.name}: {e}")
                txt_skipped += 1
    
    return npz_copied, npz_skipped_no_match, npz_skipped_multi_match, total_count, txt_copied, txt_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Copy npz files from processed directory back to original output directory, "
        "matching them with existing webp files using flexible prefix matching. "
        "NPZ files are renamed to include the random suffix from target webp files. "
        "Files with multiple matches are skipped. "
        "TXT files are always copied with renamed filenames."
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
    
    npz_copied, npz_skipped_no_match, npz_skipped_multi_match, total_npz, txt_copied, txt_skipped = copy_npz_with_structure(
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
    print(f"  Skipped (no match): {npz_skipped_no_match}")
    print(f"  Skipped (multiple matches): {npz_skipped_multi_match}")
    
    if total_npz > 0:
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

