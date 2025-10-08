#!/usr/bin/env python3
"""
Test script for astra_dataset_dropout_rate with tag preservation functionality
测试 astra_dataset_dropout_rate 保留特定标签的功能
"""

def test_astra_dropout_with_preserved_tags():
    """Test dropout with astra prefix format and preserved tags"""
    caption = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags. <Prompt Start> \nDrawn by @1-gou (111touban), masterpiece, 1girl, 2boys, best quality, 3girls, year 2024"
    
    # Tags to preserve
    preserve_tags = [
        "newest", "mid", "early", "oldest",
        "year 2023", "year 2024", "year 2025",
        "masterpiece", "best quality", "bad quality", "worst quality"
    ]
    
    # Simulate dropout logic
    astra_prefix = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags."
    
    if caption.startswith(astra_prefix):
        drawn_by_idx = caption.find("Drawn by")
        if drawn_by_idx != -1:
            comma_idx = caption.find(",", drawn_by_idx)
            if comma_idx != -1:
                kept_part = caption[:comma_idx]
                dropped_part = caption[comma_idx + 1:]
                
                # Extract preserved tags
                preserved_tags_found = []
                if dropped_part:
                    tags = [tag.strip() for tag in dropped_part.split(",")]
                    for tag in tags:
                        if tag.lower() in [pt.lower() for pt in preserve_tags]:
                            preserved_tags_found.append(tag)
                
                # Reconstruct caption
                if preserved_tags_found:
                    result = kept_part + ", " + ", ".join(preserved_tags_found)
                else:
                    result = kept_part
                
                print("✓ Test with astra prefix and preserved tags:")
                print(f"  Original: {caption}")
                print(f"  After dropout: {result}")
                print(f"  Preserved tags: {preserved_tags_found}")
                print()
                
                # Verify that quality tags are preserved
                assert "masterpiece" in result, "masterpiece should be preserved"
                assert "best quality" in result, "best quality should be preserved"
                assert "year 2024" in result, "year 2024 should be preserved"
                # Verify that non-preserved tags are removed
                assert "1girl" not in result, "1girl should be removed"
                assert "2boys" not in result, "2boys should be removed"
                assert "3girls" not in result, "3girls should be removed"

def test_astra_dropout_no_preserved_tags():
    """Test dropout when no preserved tags exist"""
    caption = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags. <Prompt Start> \nDrawn by artist_name, 1girl, solo, smile, long hair"
    
    preserve_tags = [
        "newest", "mid", "early", "oldest",
        "year 2023", "year 2024", "year 2025",
        "masterpiece", "best quality", "bad quality", "worst quality"
    ]
    
    astra_prefix = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags."
    
    if caption.startswith(astra_prefix):
        drawn_by_idx = caption.find("Drawn by")
        if drawn_by_idx != -1:
            comma_idx = caption.find(",", drawn_by_idx)
            if comma_idx != -1:
                kept_part = caption[:comma_idx]
                dropped_part = caption[comma_idx + 1:]
                
                preserved_tags_found = []
                if dropped_part:
                    tags = [tag.strip() for tag in dropped_part.split(",")]
                    for tag in tags:
                        if tag.lower() in [pt.lower() for pt in preserve_tags]:
                            preserved_tags_found.append(tag)
                
                if preserved_tags_found:
                    result = kept_part + ", " + ", ".join(preserved_tags_found)
                else:
                    result = kept_part
                
                print("✓ Test without preserved tags:")
                print(f"  Original: {caption}")
                print(f"  After dropout: {result}")
                print(f"  Preserved tags: {preserved_tags_found}")
                print()
                
                # No preserved tags should be in the result
                assert len(preserved_tags_found) == 0, "Should have no preserved tags"
                assert result == kept_part, "Result should only contain kept part"

def test_astra_dropout_without_prefix():
    """Test dropout without astra prefix format"""
    caption = "Some prompt\nDrawn by artist\nnewest, 1girl, masterpiece, 2boys, worst quality"
    
    preserve_tags = [
        "newest", "mid", "early", "oldest",
        "year 2023", "year 2024", "year 2025",
        "masterpiece", "best quality", "bad quality", "worst quality"
    ]
    
    astra_prefix = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags."
    
    if not caption.startswith(astra_prefix):
        drawn_by_idx = caption.find("Drawn by")
        if drawn_by_idx != -1:
            newline_idx = caption.find("\n", drawn_by_idx)
            if newline_idx != -1:
                kept_part = caption[:newline_idx]
                dropped_part = caption[newline_idx + 1:]
                
                preserved_tags_found = []
                if dropped_part:
                    tags = [tag.strip() for tag in dropped_part.split(",")]
                    for tag in tags:
                        if tag.lower() in [pt.lower() for pt in preserve_tags]:
                            preserved_tags_found.append(tag)
                
                if preserved_tags_found:
                    result = kept_part + "\n" + ", ".join(preserved_tags_found)
                else:
                    result = kept_part
                
                print("✓ Test without astra prefix (newline separator):")
                print(f"  Original: {caption}")
                print(f"  After dropout: {result}")
                print(f"  Preserved tags: {preserved_tags_found}")
                print()
                
                # Verify preserved tags
                assert "newest" in result, "newest should be preserved"
                assert "masterpiece" in result, "masterpiece should be preserved"
                assert "worst quality" in result, "worst quality should be preserved"
                # Verify removed tags
                assert "1girl" not in result, "1girl should be removed"
                assert "2boys" not in result, "2boys should be removed"

def test_case_insensitive():
    """Test that tag matching is case-insensitive"""
    caption = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags. <Prompt Start> \nDrawn by artist, MASTERPIECE, Best Quality, Year 2024, 1girl"
    
    preserve_tags = [
        "newest", "mid", "early", "oldest",
        "year 2023", "year 2024", "year 2025",
        "masterpiece", "best quality", "bad quality", "worst quality"
    ]
    
    astra_prefix = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags."
    
    if caption.startswith(astra_prefix):
        drawn_by_idx = caption.find("Drawn by")
        if drawn_by_idx != -1:
            comma_idx = caption.find(",", drawn_by_idx)
            if comma_idx != -1:
                kept_part = caption[:comma_idx]
                dropped_part = caption[comma_idx + 1:]
                
                preserved_tags_found = []
                if dropped_part:
                    tags = [tag.strip() for tag in dropped_part.split(",")]
                    for tag in tags:
                        if tag.lower() in [pt.lower() for pt in preserve_tags]:
                            preserved_tags_found.append(tag)
                
                if preserved_tags_found:
                    result = kept_part + ", " + ", ".join(preserved_tags_found)
                else:
                    result = kept_part
                
                print("✓ Test case-insensitive matching:")
                print(f"  Original: {caption}")
                print(f"  After dropout: {result}")
                print(f"  Preserved tags: {preserved_tags_found}")
                print()
                
                # All three should be preserved despite different cases
                assert len(preserved_tags_found) == 3, f"Should preserve 3 tags, but got {len(preserved_tags_found)}"
                assert "MASTERPIECE" in result, "MASTERPIECE should be preserved"
                assert "Best Quality" in result, "Best Quality should be preserved"
                assert "Year 2024" in result, "Year 2024 should be preserved"

if __name__ == "__main__":
    print("Testing astra_dataset_dropout_rate with tag preservation\n")
    print("=" * 80)
    print()
    
    test_astra_dropout_with_preserved_tags()
    test_astra_dropout_no_preserved_tags()
    test_astra_dropout_without_prefix()
    test_case_insensitive()
    
    print("=" * 80)
    print("\n✓ All tests passed!")
    print("\nPreserved tags:")
    print("  - newest, mid, early, oldest")
    print("  - year 2023, year 2024, year 2025")
    print("  - masterpiece, best quality, bad quality, worst quality")

