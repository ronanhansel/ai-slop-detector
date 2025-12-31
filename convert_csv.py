import pandas as pd
import sys
import os

def convert_pkl_to_csv(pkl_file, csv_file=None):
    """
    Convert a pickle file containing a pandas DataFrame to CSV
    
    Args:
        pkl_file: Path to the pickle file
        csv_file: Output CSV file path (optional, defaults to same name with .csv extension)
    """
    # Read the pickle file
    df = pd.read_pickle(pkl_file)
    
    # If no output file specified, use the same name with .csv extension
    if csv_file is None:
        csv_file = os.path.splitext(pkl_file)[0] + '.csv'
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"✓ Converted {pkl_file} to {csv_file}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return csv_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv.py <pickle_file> [output_csv_file]")
        print("\nExample:")
        print("  python convert_csv.py data.pkl")
        print("  python convert_csv.py data.pkl output.csv")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pkl_file):
        print(f"Error: File '{pkl_file}' not found")
        sys.exit(1)
    
    convert_pkl_to_csv(pkl_file, csv_file)
