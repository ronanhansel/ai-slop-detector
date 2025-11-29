import json
import pandas as pd
from pathlib import Path


def flatten_tweets(data, parent_id=None, source_file=None):
    """
    Recursively flatten nested tweet structure into a list of rows.
    
    Args:
        data: List of tweet objects or a single tweet object
        parent_id: The link/id of the parent tweet (for replies)
        source_file: The name of the source JSON file (influencer username)
    
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
            'replies': tweet.get('replies', 0),
            'source_influencer': source_file or ''  # Track which influencer's data this came from
        }
        
        rows.append(row)
        
        # Recursively process replies if they exist
        if 'replies_content' in tweet and tweet['replies_content']:
            current_tweet_id = tweet.get('link', '')
            replies_rows = flatten_tweets(tweet['replies_content'], parent_id=current_tweet_id, source_file=source_file)
            rows.extend(replies_rows)
    
    return rows


def process_all_json_files(data_dir, output_pickle_path):
    """
    Process all JSON files in the influencer_data directory and combine into one DataFrame.
    
    Args:
        data_dir: Path to the directory containing influencer_data folder
        output_pickle_path: Path to save the output pickle file
    """
    influencer_data_dir = Path(data_dir) / 'influencer_data'
    
    if not influencer_data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {influencer_data_dir}")
    
    all_rows = []
    processed_files = 0
    failed_files = []
    
    # Get all JSON files
    json_files = list(influencer_data_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            # Extract influencer name from filename (without .json extension)
            influencer_name = json_file.stem
            
            print(f"Processing {influencer_name}...")
            
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten the nested structure
            rows = flatten_tweets(data, source_file=influencer_name)
            all_rows.extend(rows)
            
            processed_files += 1
            print(f"  ✓ Added {len(rows)} tweets from {influencer_name}")
            
        except Exception as e:
            failed_files.append((json_file.name, str(e)))
            print(f"  ✗ Error processing {json_file.name}: {e}")
    
    # Create DataFrame from all rows
    df = pd.DataFrame(all_rows)
    
    # Save as pickle
    output_path = Path(output_pickle_path)
    df.to_pickle(output_path)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}/{len(json_files)}")
    print(f"Total tweets: {len(df)}")
    print(f"Failed files: {len(failed_files)}")
    if failed_files:
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Display some statistics
    print(f"\nDataFrame Info:")
    print(f"  - Unique influencers: {df['source_influencer'].nunique()}")
    print(f"  - Unique usernames: {df['username'].nunique()}")
    print(f"  - Top-level tweets: {(df['parent_id'] == '').sum()}")
    print(f"  - Reply tweets: {(df['parent_id'] != '').sum()}")
    
    return df


if __name__ == '__main__':
    import sys
    
    output_dir = Path('../data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default paths
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data-ai-slop-detector'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '../data/all_tweets.pkl'
    
    try:
        df = process_all_json_files(data_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
