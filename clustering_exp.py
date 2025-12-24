import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_PERCENTAGE = 0.05  # 5% by default - adjust as needed

# Load the processed pickle file
with open('data-ai-slop-detector/final_detection_processed.pkl', 'rb') as f:
    df = pd.read_pickle(f)

print(f"Loaded processed dataframe with shape: {df.shape}")
print(f"Using {SAMPLE_PERCENTAGE*100}% of data for analysis")

# # Prepare data splits based on confidence levels
# df_full = df.dropna(subset=['label', 'ai_confidence', 'minilm_embedding', 'tf_idf_embedding', 'empath_embedding']).copy()

# # Sample data for computational efficiency
# df_full_sampled = df_full.sample(frac=SAMPLE_PERCENTAGE, random_state=42)

# # Create confidence-based subsets from the full data (not sampled yet)
# df_conf_07 = df_full[(df_full['ai_confidence'] >= 0.7) | (df_full['ai_confidence'] <= 0.3)].copy()
# df_conf_09 = df_full[(df_full['ai_confidence'] >= 0.9) | (df_full['ai_confidence'] <= 0.1)].copy()

# # Sample the confidence subsets
# df_conf_07_sampled = df_conf_07.sample(frac=SAMPLE_PERCENTAGE, random_state=42) if len(df_conf_07) > 100 else df_conf_07
# df_conf_09_sampled = df_conf_09.sample(frac=SAMPLE_PERCENTAGE, random_state=42) if len(df_conf_09) > 100 else df_conf_09

# print(f"\nDataset sizes (original -> sampled):")
# print(f"Full data: {len(df_full)} -> {len(df_full_sampled)}")
# print(f"Confidence >= 0.7 or <= 0.3: {len(df_conf_07)} -> {len(df_conf_07_sampled)}")
# print(f"Confidence >= 0.9 or <= 0.1: {len(df_conf_09)} -> {len(df_conf_09_sampled)}")

def balanced_sample(df, label_col='label', n_per_class=5000, random_state=42):
    """Sample up to n_per_class from each class for balanced dataset."""
    # If label is not numeric, map to 0/1
    y_raw = df[label_col]
    if y_raw.dtype.kind in {'U', 'S', 'O'}:
        unique_labels = sorted(np.unique(y_raw))
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        df = df.copy()
        df[label_col] = df[label_col].map(label_map)
    # Sample
    grouped = df.groupby(label_col, group_keys=False)
    sampled = grouped.apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state))
    return sampled

# Prepare data splits based on confidence levels
df_full = df.dropna(subset=['label', 'ai_confidence', 'minilm_embedding', 'tf_idf_embedding', 'empath_embedding']).copy()

# Balanced sampling for each dataset
df_full_sampled = balanced_sample(df_full, label_col='label', n_per_class=5000, random_state=42)
df_conf_07 = df_full[(df_full['ai_confidence'] >= 0.7) | (df_full['ai_confidence'] <= 0.3)].copy()
df_conf_07_sampled = balanced_sample(df_conf_07, label_col='label', n_per_class=5000, random_state=42) if len(df_conf_07) > 0 else df_conf_07
df_conf_09 = df_full[(df_full['ai_confidence'] >= 0.9) | (df_full['ai_confidence'] <= 0.1)].copy()
df_conf_09_sampled = balanced_sample(df_conf_09, label_col='label', n_per_class=5000, random_state=42) if len(df_conf_09) > 0 else df_conf_09

print(f"\nDataset sizes (original -> balanced sampled):")
print(f"Full data: {len(df_full)} -> {len(df_full_sampled)}")
print(f"Confidence >= 0.7 or <= 0.3: {len(df_conf_07)} -> {len(df_conf_07_sampled)}")
print(f"Confidence >= 0.9 or <= 0.1: {len(df_conf_09)} -> {len(df_conf_09_sampled)}")

# Function to perform clustering and visualization
def cluster_and_visualize(data, embedding_col, dataset_name, output_prefix):
    """
    Perform KMeans clustering on embeddings and visualize results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name} - {embedding_col}")
    print(f"{'='*60}")
    
    # Extract embeddings as numpy array
    X = np.vstack(data[embedding_col].values)
    # Convert y_true to 0/1 if not already numeric
    y_true_raw = data['label'].values
    if y_true_raw.dtype.kind in {'U', 'S', 'O'}:
        # Map unique labels to 0/1
        unique_labels = sorted(np.unique(y_true_raw))
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        y_true = np.array([label_map[x] for x in y_true_raw])
    else:
        y_true = y_true_raw.astype(int)
    
    print(f"Embedding shape: {X.shape}")
    print(f"True labels distribution: {pd.Series(y_true).value_counts().to_dict()}")
    
    # Perform KMeans clustering (k=2 for binary classification)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    
    # Align cluster labels with true labels (find best mapping)
    cm = confusion_matrix(y_true, y_pred)
    if cm[0, 1] + cm[1, 0] > cm[0, 0] + cm[1, 1]:
        # Swap cluster labels if needed
        y_pred = 1 - y_pred
    
    print(f"\nMetrics:")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"\nConfusion Matrix:")
    cm_final = confusion_matrix(y_true, y_pred)
    print(cm_final)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
    
    # PCA for visualization (reduce to 2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: True labels
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=15)
    axes[0].set_title(f'True Labels\n{dataset_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter1, ax=axes[0], label='True Label')
    
    # Plot 2: Predicted clusters
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='plasma', alpha=0.6, s=15)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
    axes[1].set_title(f'KMeans Clusters\nARI: {ari:.4f}, Silhouette: {silhouette:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1].legend()
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    # Plot 3: Confusion matrix heatmap
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', ax=axes[2], 
                xticklabels=['Cluster 0', 'Cluster 1'],
                yticklabels=['True 0', 'True 1'])
    axes[2].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Cluster')
    
    plt.tight_layout()
    output_file = f'data-ai-slop-detector/{output_prefix}_{embedding_col}_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.show()
    
    return {
        'dataset': dataset_name,
        'embedding': embedding_col,
        'n_samples': len(data),
        'ari': ari,
        'silhouette': silhouette,
        'confusion_matrix': cm_final.tolist(),
        'accuracy': (cm_final[0,0] + cm_final[1,1]) / cm_final.sum()
    }

# Run clustering experiments
embeddings = ['minilm_embedding', 'tf_idf_embedding', 'empath_embedding']
datasets = {
    'Full_Data': df_full_sampled,
    'Conf_0.7': df_conf_07_sampled,
    'Conf_0.9': df_conf_09_sampled
}

results = []

for dataset_name, dataset in datasets.items():
    print(f"\n{'#'*80}")
    print(f"# {dataset_name.replace('_', ' ')} ({len(dataset)} samples)")
    print(f"{'#'*80}")
    
    for embedding_col in embeddings:
        result = cluster_and_visualize(dataset, embedding_col, dataset_name, 'kmeans_clustering')
        results.append(result)

# Create summary table
results_df = pd.DataFrame(results)
print(f"\n{'='*80}")
print("SUMMARY OF CLUSTERING RESULTS")
print(f"{'='*80}\n")
print(results_df[['dataset', 'embedding', 'n_samples', 'ari', 'silhouette', 'accuracy']].to_string(index=False))

# Save summary
results_df.to_csv('data-ai-slop-detector/kmeans_clustering_summary.csv', index=False)
print(f"\nSaved summary to: data-ai-slop-detector/kmeans_clustering_summary.csv")

# Create comparative visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for i, embedding_col in enumerate(embeddings):
    for j, (dataset_name, dataset) in enumerate(datasets.items()):
        ax = axes[j, i]
        
        # Get subset of results
        subset = results_df[(results_df['embedding'] == embedding_col) & 
                           (results_df['dataset'] == dataset_name)]
        
        if not subset.empty:
            ari = subset['ari'].values[0]
            silhouette = subset['silhouette'].values[0]
            accuracy = subset['accuracy'].values[0]
            n_samples = subset['n_samples'].values[0]
            
            # Create bar chart
            metrics = ['ARI', 'Silhouette', 'Accuracy']
            values = [ari, silhouette, accuracy]
            colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in values]
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim(-0.1, 1.0)
            ax.set_title(f'{embedding_col.replace("_embedding", "").upper()}\n{dataset_name} (n={n_samples})', 
                        fontsize=10, fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle(f'KMeans Clustering Performance Comparison ({SAMPLE_PERCENTAGE*100}% sample)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('data-ai-slop-detector/kmeans_clustering_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved comparison chart to: data-ai-slop-detector/kmeans_clustering_comparison.png")
plt.show()

# Create a detailed comparison heatmap
pivot_ari = results_df.pivot(index='embedding', columns='dataset', values='ari')
pivot_silhouette = results_df.pivot(index='embedding', columns='dataset', values='silhouette')
pivot_accuracy = results_df.pivot(index='embedding', columns='dataset', values='accuracy')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(pivot_ari, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0], 
            vmin=0, vmax=1, cbar_kws={'label': 'ARI'})
axes[0].set_title('Adjusted Rand Index', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dataset')
axes[0].set_ylabel('Embedding Type')

sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
            vmin=0, vmax=1, cbar_kws={'label': 'Silhouette'})
axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dataset')
axes[1].set_ylabel('')

sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[2],
            vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
axes[2].set_title('Clustering Accuracy', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Dataset')
axes[2].set_ylabel('')

plt.suptitle(f'Clustering Performance Metrics Heatmaps ({SAMPLE_PERCENTAGE*100}% sample)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data-ai-slop-detector/kmeans_clustering_heatmaps.png', dpi=150, bbox_inches='tight')
print(f"\nSaved heatmaps to: data-ai-slop-detector/kmeans_clustering_heatmaps.png")
plt.show()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\nTotal experiments run: {len(results)}")
print(f"Sample size: {SAMPLE_PERCENTAGE*100}% of original data")
print(f"\nTo adjust sample size, change SAMPLE_PERCENTAGE at the top of the script")