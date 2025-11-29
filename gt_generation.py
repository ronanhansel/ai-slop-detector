import pandas as pd
import numpy as np
import sys
import os
import random
from tqdm import tqdm
import torch

# Add helper directory to path to import detect.py
# Assuming notebook is in ai-slop-detector/
sys.path.append(os.path.abspath('helper'))

# Import the detection function
# Note: This will load the model (which might take a moment)
try:
    from detect import detect
    print("Detection module imported successfully.")
except ImportError as e:
    print(f"Error importing detect module: {e}")
    print("Make sure 'helper/detect.py' exists and required packages are installed.")
    
    
# Configuration
DATA_DIR = 'data-ai-slop-detector'
# Prefer the NLP merged file if available, else processed comments
MERGED_FILE = os.path.join(DATA_DIR, 'final_merged_data_nlp.pkl')
COMMENTS_FILE = os.path.join(DATA_DIR, 'processed_comments.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'final_detection.pkl')

# Load Data
if os.path.exists(OUTPUT_FILE):
    print(f"Found existing progress file: {OUTPUT_FILE}")
    df = pd.read_pickle(OUTPUT_FILE)
    print("Loaded data from detection.pkl")
elif os.path.exists(MERGED_FILE):
    print(f"Loading from merged NLP data: {MERGED_FILE}")
    df = pd.read_pickle(MERGED_FILE)
else:
    print(f"Loading from processed comments: {COMMENTS_FILE}")
    df = pd.read_csv(COMMENTS_FILE)

# Initialize columns if they don't exist
if 'label' not in df.columns:
    print("Initializing 'label' column...")
    df['label'] = None
if 'ai_confidence' not in df.columns:
    print("Initializing 'ai_confidence' column...")
    df['ai_confidence'] = np.nan

total_rows = len(df)
labeled_rows = df['label'].notna().sum()
remaining_rows = total_rows - labeled_rows

print("-" * 30)
print(f"Total rows:       {total_rows}")
print(f"Already labeled:  {labeled_rows}")
print(f"Remaining:        {remaining_rows}")
print("-" * 30)

if labeled_rows > 0:
    print("RESUMING detection task...")
else:
    print("STARTING detection task...")

# Parameters
SAVE_EVERY_N_ROWS = 100

# Filter for rows that need labeling
unlabeled_indices = df[df['label'].isna()].index
print(f"Processing {len(unlabeled_indices)} unlabeled rows...")

# Counter for saving
rows_processed_since_save = 0

try:
    for idx in tqdm(unlabeled_indices, desc="Labeling Comments"):
        text = df.loc[idx, 'comment_content']
        
        # Basic validation
        if not isinstance(text, str) or not text.strip():
            # Mark empty/invalid text as skipped (or handle as you prefer)
            # For now, we just skip detection but don't label it, 
            # so it might be processed again. 
            # To avoid reprocessing, we could set a special label like 'SKIPPED'
            # df.at[idx, 'label'] = 'SKIPPED' 
            continue
            
        try:
            # Run detection
            result = detect(text)
            label = result['prediction_label'] # 'human' or 'AI'
            confidence = result['ai_prob']
            
            # Update DataFrame
            df.at[idx, 'label'] = label
            df.at[idx, 'ai_confidence'] = confidence
            
        except Exception as e:
            print(f"Error detecting index {idx}: {e}")
            continue
        
        # Save progress
        rows_processed_since_save += 1
        if rows_processed_since_save >= SAVE_EVERY_N_ROWS:
            df.to_pickle(OUTPUT_FILE)
            rows_processed_since_save = 0
            
except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving progress...")
finally:
    # Final save on completion or interruption
    df.to_pickle(OUTPUT_FILE)
    print(f"Progress saved to {OUTPUT_FILE}")
    print(f"Total labeled now: {df['label'].notna().sum()}")