import json
import re
from pathlib import Path
import csv
from typing import Dict, List, Tuple
from process_posts import (
    count_emojis, count_text_emojis, count_caps_locked_words,
    count_unicode_chars, contains_media_link, contains_link,
    count_tagged_people, tagged_grok, used_slang_or_abbreviation,
    clean_text_for_posts_Empath, clean_text_for_posts_LSA
)

# Global counter for unique comment IDs across all files
global_comment_id = 0


def process_comments_recursive(
    comment: Dict,
    post_author: str,
    post_id: int,
    parent_id: int = None,
    parent_author: str = None
) -> List[Dict]:
    """
    Recursively process comments and their replies.
    
    Args:
        comment: Comment dictionary
        post_author: Author of the post
        post_id: ID of the post
        parent_id: ID of parent comment (None for root comments)
        parent_author: Username of parent comment author
        
    Returns:
        List of comment dictionaries
    """
    global global_comment_id
    global_comment_id += 1
    current_comment_id = global_comment_id
    
    username = comment.get("username", "")
    original_content = comment.get("content", "")
    
    # Extract metadata before cleaning
    num_emojis = count_emojis(original_content)
    num_text_emojis = count_text_emojis(original_content)
    num_caps_words = count_caps_locked_words(original_content)
    num_unicode_chars = count_unicode_chars(original_content)
    has_media = contains_media_link(original_content)
    has_link = contains_link(original_content)
    
    # Count tagged people (exclude post author and parent author)
    exclude_list = [post_author]
    if parent_author:
        exclude_list.append(parent_author)
    num_tagged = count_tagged_people(original_content, exclude_authors=exclude_list)
    
    did_tag_grok = tagged_grok(original_content)
    
    # Clean content
    cleaned_content_LSA = clean_text_for_posts_LSA(original_content, author=username)
    clean_content_Empath = clean_text_for_posts_Empath(original_content, author=username)
    
    # Check for slang usage
    used_slang = used_slang_or_abbreviation(original_content, cleaned_content_LSA)
    
    # Create comment row
    comment_row = {
        "commenter_id": username,
        "comment_id": current_comment_id,
        "parent_id": parent_id if parent_id else "",
        "post_id": post_id,
        "comment_content": original_content,
        "cleaned_content_LSA": cleaned_content_LSA,
        "cleaned_content_Empath": clean_content_Empath,
        "num_emojis": num_emojis,
        "num_text_emojis": num_text_emojis,
        "num_caps_words": num_caps_words,
        "num_unicode_chars": num_unicode_chars,
        "contains_media": has_media,
        "contains_link": has_link,
        "num_tagged_people": num_tagged,
        "tagged_grok": did_tag_grok,
        "used_slang": used_slang
    }
    
    rows = [comment_row]
    
    # Process nested replies
    if "replies_content" in comment and comment["replies_content"]:
        for reply in comment["replies_content"]:
            nested_rows = process_comments_recursive(
                reply,
                post_author=post_author,
                post_id=post_id,
                parent_id=current_comment_id,
                parent_author=username
            )
            rows.extend(nested_rows)
    
    return rows

def process_single_file_comments(json_file_path: str) -> List[Dict]:
    """
    Process comments from a single JSON file.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        List of comment dictionaries
    """
    json_file = Path(json_file_path)
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_comments = []
    
    for post_id, post in enumerate(data, start=1):
        post_author = post.get("username", "")
        
        # Process comments for this post
        if "replies_content" in post and post["replies_content"]:
            for comment in post["replies_content"]:
                comment_rows = process_comments_recursive(
                    comment,
                    post_author=post_author,
                    post_id=post_id,
                    parent_id=None,
                    parent_author=post_author
                )
                all_comments.extend(comment_rows)
    
    return all_comments

def process_all_comments(input_dir: str, output_csv_path: str) -> None:
    """
    Process all comment JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_csv_path: Path to output CSV file
    """
    global global_comment_id
    global_comment_id = 0  # Reset counter
    
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    all_comments = []
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        comments = process_single_file_comments(json_file)
        all_comments.extend(comments)
        print(f"  Extracted {len(comments)} comments")
    
    # Write all to CSV
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "commenter_id", "comment_id", "parent_id", "post_id",
        "comment_content", "cleaned_content_LSA", "cleaned_content_Empath", "num_emojis",
        "num_text_emojis", "num_caps_words", "num_unicode_chars",
        "contains_media", "contains_link", "num_tagged_people",
        "tagged_grok", "used_slang"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comments)
    
    print(f"\n✅ Total comments processed: {len(all_comments)}")
    print(f"   Saved to: {output_file}")
    print(f"   Unique comment IDs: 1 to {global_comment_id}")

if __name__ == "__main__":
    # Example usage
    input_directory = r"E:\Project_DS\ai-slop-detector\data_preparation\influencer_data"
    output_csv = "data_preparation/outputs/processed_comments.csv"
    process_all_comments(input_directory, output_csv)