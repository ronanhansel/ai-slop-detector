import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PREPARING DATA FOR CLASSIFIER")
print("="*80)

# Load processed data (has minilm_embedding)
print("\n1. Loading processed data with MiniLM embeddings...")
with open('data-ai-slop-detector/final_detection_processed.pkl', 'rb') as f:
    df_processed = pd.read_pickle(f)
print(f"Loaded processed data: {df_processed.shape}")

# Load original data (has individual empath columns)
print("\n2. Loading original data with Empath features...")
with open('data-ai-slop-detector/final_detection.pkl', 'rb') as f:
    df_original = pd.read_pickle(f)
print(f"Loaded original data: {df_original.shape}")

# Define features
behavioral_features = [
    'num_emojis', 'num_text_emojis', 'num_caps_words', 'num_unicode_chars',
    'contains_media', 'contains_link', 'num_tagged_people', 'tagged_grok', 'used_slang'
]

empath_features_selected = [
    'internet', 'swearing_terms', 'positive_emotion', 'negative_emotion',
    'technology', 'speaking', 'celebration'
]

print(f"\n3. Extracting features...")
print(f"   Behavioral features: {len(behavioral_features)}")
print(f"   Empath features: {len(empath_features_selected)}")

# Merge datasets on index to get all features
print("\n4. Merging datasets...")
df_merged = df_processed.copy()

# Add empath features from original
for feat in empath_features_selected:
    if feat in df_original.columns:
        df_merged[feat] = df_original[feat]

# Keep only rows with all required data
required_cols = ['label', 'ai_confidence', 'minilm_embedding'] + behavioral_features + empath_features_selected
df_clean = df_merged.dropna(subset=required_cols).copy()

print(f"Data after merging and cleaning: {len(df_clean)} rows")
print(f"Label distribution: {df_clean['label'].value_counts().to_dict()}")

# ===== COMPUTE MINILM CLUSTER CENTROIDS =====
print("\n" + "="*80)
print("COMPUTING MINILM CENTROIDS (K=2)")
print("="*80)

# Extract MiniLM embeddings
X_minilm = np.vstack(df_clean['minilm_embedding'].values)
print(f"MiniLM embedding shape: {X_minilm.shape}")

# Convert labels to numeric (0/1)
y_true_raw = df_clean['label'].values
if y_true_raw.dtype.kind in {'U', 'S', 'O'}:
    unique_labels = sorted(np.unique(y_true_raw))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y_numeric = np.array([label_map[x] for x in y_true_raw])
    print(f"Label mapping: {label_map}")
else:
    y_numeric = y_true_raw.astype(int)

print(f"Numeric label distribution: {pd.Series(y_numeric).value_counts().to_dict()}")

# Perform KMeans clustering
print("\nPerforming KMeans clustering (k=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_minilm)
centroids = kmeans.cluster_centers_

print(f"Cluster 0 centroid shape: {centroids[0].shape}")
print(f"Cluster 1 centroid shape: {centroids[1].shape}")
print(f"Cluster assignments: {pd.Series(cluster_labels).value_counts().to_dict()}")

# Calculate distances to both centroids for each sample
print("\nCalculating distances to centroids...")
dist_to_centroid_0 = np.linalg.norm(X_minilm - centroids[0], axis=1)
dist_to_centroid_1 = np.linalg.norm(X_minilm - centroids[1], axis=1)

df_clean['dist_to_centroid_0'] = dist_to_centroid_0
df_clean['dist_to_centroid_1'] = dist_to_centroid_1

print(f"Distance statistics:")
print(f"  Dist to Centroid 0: mean={dist_to_centroid_0.mean():.4f}, std={dist_to_centroid_0.std():.4f}")
print(f"  Dist to Centroid 1: mean={dist_to_centroid_1.mean():.4f}, std={dist_to_centroid_1.std():.4f}")

# Save centroids for future use
centroids_data = {
    'centroids': centroids,
    'label_map': label_map if y_true_raw.dtype.kind in {'U', 'S', 'O'} else None,
    'kmeans_model': kmeans
}

with open('data-ai-slop-detector/minilm_centroids.pkl', 'wb') as f:
    pickle.dump(centroids_data, f)
print("\nSaved centroids to: data-ai-slop-detector/minilm_centroids.pkl")

# ===== PREPARE FINAL FEATURE SET =====
print("\n" + "="*80)
print("PREPARING FINAL FEATURE SET")
print("="*80)

feature_columns = behavioral_features + empath_features_selected + ['dist_to_centroid_0', 'dist_to_centroid_1']
print(f"\nTotal features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# Create final dataset with all features
df_final = df_clean[['label', 'ai_confidence'] + feature_columns].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Missing values per column:")
print(df_final.isnull().sum())

# Save prepared data
output_path = 'data-ai-slop-detector/classifier_data_prepared.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(df_final, f)

print(f"\n{'='*80}")
print("DATA PREPARATION COMPLETE")
print(f"{'='*80}")
print(f"\nSaved prepared data to: {output_path}")
print(f"Total samples: {len(df_final)}")
print(f"Total features: {len(feature_columns)}")
print(f"Feature breakdown:")
print(f"  - Behavioral: {len(behavioral_features)}")
print(f"  - Empath: {len(empath_features_selected)}")
print(f"  - MiniLM distances: 2")