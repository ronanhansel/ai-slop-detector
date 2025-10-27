import json
import csv
from pathlib import Path


def flatten_tweets(data, parent_id=None):
    """
    Recursively flatten nested tweet structure into a list of rows.
    
    Args:
        data: List of tweet objects or a single tweet object
        parent_id: The link/id of the parent tweet (for replies)
    
    Returns:
        List of flattened tweet dictionaries
    """
    rows = []
    
    # Handle both single object and list
    if isinstance(data, dict):
        data = [data]
    
    for tweet in data:
        # Skip empty objects
        if not tweet:
            continue
            
        # Extract main tweet data
        row = {
            'id': tweet.get('link', ''),
            'username': tweet.get('username', ''),
            'content': tweet.get('content', ''),
            'parent_id': parent_id or '',  # Empty string if no parent
            'likes': tweet.get('likes', 0),
            'retweet': tweet.get('retweet', 0),
            'share': tweet.get('share', 0),
            'views': tweet.get('views', 0),
            'replies': tweet.get('replies', 0)
        }
        
        rows.append(row)
        
        # Recursively process replies if they exist
        if 'replies_content' in tweet and tweet['replies_content']:
            current_tweet_id = tweet.get('link', '')
            replies_rows = flatten_tweets(tweet['replies_content'], parent_id=current_tweet_id)
            rows.extend(replies_rows)
    
    return rows


def json_to_csv(json_file_path, csv_file_path=None):
    """
    Convert nested JSON tweet data to flat CSV format.
    
    Args:
        json_file_path: Path to input JSON file
        csv_file_path: Path to output CSV file (optional, defaults to same name with .csv)
    """
    # Read JSON file
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten the nested structure
    rows = flatten_tweets(data)
    
    # Determine output CSV path
    if csv_file_path is None:
        csv_file_path = json_path.with_suffix('.csv')
    
    csv_path = Path(csv_file_path)
    
    # Write to CSV
    fieldnames = ['id', 'username', 'content', 'parent_id', 'likes', 'retweet', 'share', 'views', 'replies']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully converted {len(rows)} tweets to CSV")
    print(f"Output file: {csv_path}")
    
    return csv_path


if __name__ == '__main__':
    import sys
    
    # Default to ByYourLogic.json in data folder
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../data/ByYourLogic.json'
    
    # Optional output file path
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        json_to_csv(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
