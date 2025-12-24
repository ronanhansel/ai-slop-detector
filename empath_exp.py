import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# Load the pickle file
with open('data-ai-slop-detector/final_detection.pkl', 'rb') as f:
    df = pd.read_pickle(f)

print(f"Loaded dataframe with shape: {df.shape}")

# Define Empath columns
empath_cols = [
    'help', 'office', 'dance', 'money', 'wedding', 'domestic_work', 'sleep',
    'medical_emergency', 'cold', 'hate', 'cheerfulness', 'aggression', 'occupation',
    'envy', 'anticipation', 'family', 'vacation', 'crime', 'attractive', 'masculine',
    'prison', 'health', 'pride', 'dispute', 'nervousness', 'government', 'weakness',
    'horror', 'swearing_terms', 'leisure', 'suffering', 'royalty', 'wealthy', 'tourism',
    'furniture', 'school', 'magic', 'beach', 'journalism', 'morning', 'banking',
    'social_media', 'exercise', 'night', 'kill', 'blue_collar_job', 'art', 'ridicule',
    'play', 'computer', 'college', 'optimism', 'stealing', 'real_estate', 'home',
    'divine', 'sexual', 'fear', 'irritability', 'superhero', 'business', 'driving',
    'pet', 'childish', 'cooking', 'exasperation', 'religion', 'hipster', 'internet',
    'surprise', 'reading', 'worship', 'leader', 'independence', 'movement', 'body',
    'noise', 'eating', 'medieval', 'zest', 'confusion', 'water', 'sports', 'death',
    'healing', 'legend', 'heroic', 'celebration', 'restaurant', 'violence', 'programming',
    'dominant_heirarchical', 'military', 'neglect', 'swimming', 'exotic', 'love',
    'hiking', 'communication', 'hearing', 'order', 'sympathy', 'hygiene', 'weather',
    'anonymity', 'trust', 'ancient', 'deception', 'fabric', 'air_travel', 'fight',
    'dominant_personality', 'music', 'vehicle', 'politeness', 'toy', 'farming',
    'meeting', 'war', 'speaking', 'listen', 'urban', 'shopping', 'disgust', 'fire',
    'tool', 'phone', 'gain', 'sound', 'injury', 'sailing', 'rage', 'science', 'work',
    'appearance', 'valuable', 'warmth', 'youth', 'sadness', 'fun', 'emotional', 'joy',
    'affection', 'traveling', 'fashion', 'ugliness', 'lust', 'shame', 'torment',
    'economics', 'anger', 'politics', 'ship', 'clothing', 'car', 'strength',
    'technology', 'breaking', 'shape_and_size', 'power', 'white_collar_job', 'animal',
    'party', 'terrorism', 'smell', 'disappointment', 'poor', 'plant', 'pain', 'beauty',
    'timidity', 'philosophy', 'negotiate', 'negative_emotion', 'cleaning', 'messaging',
    'competing', 'law', 'friends', 'payment', 'achievement', 'alcohol', 'liquid',
    'feminine', 'weapon', 'children', 'monster', 'ocean', 'giving', 'contentment',
    'writing', 'rural', 'positive_emotion', 'musical'
]

# Prepare data
df_clean = df.dropna(subset=['label'] + empath_cols).copy()

# Encode labels
if df_clean['label'].dtype.kind in {'U', 'S', 'O'}:
    le = LabelEncoder()
    y = le.fit_transform(df_clean['label'])
    label_names = le.classes_
    print(f"\nEncoded labels: {dict(zip(label_names, range(len(label_names))))}")
else:
    y = df_clean['label'].values
    label_names = ['Class 0', 'Class 1']

# Extract features
X = df_clean[empath_cols].values
X_df = pd.DataFrame(X, columns=empath_cols)

print(f"\nDataset info:")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

# Sample data for faster SHAP computation (use 10,000 samples)
sample_size = min(10000, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[indices]
X_sample_df = X_df.iloc[indices]
y_sample = y[indices]

print(f"\nUsing {sample_size} samples for SHAP analysis")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"Random Forest accuracy: {rf_score:.4f}")

# Train Gradient Boosting classifier
print("\nTraining Gradient Boosting classifier...")
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_test, y_test)
print(f"Gradient Boosting accuracy: {gb_score:.4f}")

# Choose the best model
best_model = rf_model if rf_score >= gb_score else gb_model
model_name = "Random Forest" if rf_score >= gb_score else "Gradient Boosting"
print(f"\nUsing {model_name} for SHAP analysis (accuracy: {max(rf_score, gb_score):.4f})")

# ===== SHAP ANALYSIS =====
print("\n" + "="*80)
print("COMPUTING SHAP VALUES")
print("="*80)

# Use a smaller background sample for faster computation
background_size = min(100, len(X_train))
background = X_train[np.random.choice(len(X_train), background_size, replace=False)]

# Create SHAP explainer
print(f"\nCreating SHAP explainer with {background_size} background samples...")
explainer = shap.TreeExplainer(best_model, background)

# Compute SHAP values for test set
print("Computing SHAP values for test set...")
shap_values = explainer.shap_values(X_test)

# Handle multi-output case (for binary classification, some models return 2 outputs)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use values for positive class

print(f"SHAP values shape: {shap_values.shape}")

# ===== FEATURE IMPORTANCE RANKING =====
print("\n" + "="*80)
print("FEATURE IMPORTANCE RANKING (SHAP)")
print("="*80)

# Calculate mean absolute SHAP value for each feature
# Handle both 2D and 3D SHAP values (binary vs multi-class)
if shap_values.ndim == 3:
    # For binary classification with 2 outputs, use the positive class
    shap_values = shap_values[:, :, 1]
elif shap_values.ndim == 2:
    # Already in correct shape
    pass

print(f"SHAP values shape after processing: {shap_values.shape}")

mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Ensure mean_abs_shap is 1D
if mean_abs_shap.ndim > 1:
    mean_abs_shap = mean_abs_shap.flatten()

print(f"Mean absolute SHAP shape: {mean_abs_shap.shape}")
print(f"Number of features: {len(empath_cols)}")

feature_importance = pd.DataFrame({
    'feature': empath_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 30 Most Important Features:")
print(feature_importance.head(30).to_string(index=False))

print("\nBottom 20 Least Important Features:")
print(feature_importance.tail(20).to_string(index=False))

# Save results
feature_importance.to_csv('data-ai-slop-detector/empath_shap_importance.csv', index=False)
print(f"\nSaved SHAP importance to: data-ai-slop-detector/empath_shap_importance.csv")

# ===== VISUALIZATIONS =====
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. SHAP Beeswarm Plot (THE GOLD STANDARD)
print("\n1. Creating SHAP Beeswarm Plot...")
plt.figure(figsize=(12, 14))
shap.summary_plot(shap_values, X_test, feature_names=empath_cols, plot_type="dot", 
                  max_display=30, show=False)
plt.title('SHAP Beeswarm Plot - Top 30 Features\n(Red = High Feature Value, Blue = Low Feature Value)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('data-ai-slop-detector/shap_beeswarm_plot.png', dpi=150, bbox_inches='tight')
print("Saved: data-ai-slop-detector/shap_beeswarm_plot.png")
plt.show()

# 2. SHAP Bar Plot (Feature Importance)
print("\n2. Creating SHAP Bar Plot...")
plt.figure(figsize=(12, 14))
shap.summary_plot(shap_values, X_test, feature_names=empath_cols, plot_type="bar", 
                  max_display=30, show=False)
plt.title('SHAP Feature Importance - Top 30 Features\n(Mean |SHAP Value|)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('data-ai-slop-detector/shap_bar_plot.png', dpi=150, bbox_inches='tight')
print("Saved: data-ai-slop-detector/shap_bar_plot.png")
plt.show()

# 3. Custom Bar Plot with Colors
print("\n3. Creating custom importance bar plot...")
fig, ax = plt.subplots(figsize=(12, 14))
top_30 = feature_importance.head(30)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_30)))
bars = ax.barh(range(len(top_30)), top_30['mean_abs_shap'].values, color=colors)
ax.set_yticks(range(len(top_30)))
ax.set_yticklabels(top_30['feature'].values)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Empath Features by SHAP Importance\n(Bot vs Human Classification)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_30['mean_abs_shap'].values)):
    ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('data-ai-slop-detector/shap_custom_bar_plot.png', dpi=150, bbox_inches='tight')
print("Saved: data-ai-slop-detector/shap_custom_bar_plot.png")
plt.show()

# 4. SHAP Heatmap for Top Features
print("\n4. Creating SHAP heatmap...")
top_20_features = feature_importance.head(20)['feature'].values
top_20_indices = [empath_cols.index(f) for f in top_20_features]
shap_values_top20 = shap_values[:, top_20_indices]

# Sort by SHAP values
sample_indices = np.argsort(shap_values_top20.sum(axis=1))[::-1][:200]  # Top 200 samples
shap_heatmap_data = shap_values_top20[sample_indices]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(shap_heatmap_data.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_yticks(range(len(top_20_features)))
ax.set_yticklabels(top_20_features, fontsize=10)
ax.set_xlabel('Samples (sorted by total SHAP impact)', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('SHAP Values Heatmap - Top 20 Features\n(Red = Pushes toward Bot, Blue = Pushes toward Human)', 
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='SHAP Value')
plt.tight_layout()
plt.savefig('data-ai-slop-detector/shap_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: data-ai-slop-detector/shap_heatmap.png")
plt.show()

# 5. Comparison with Feature Importance from Model
print("\n5. Creating model feature importance comparison...")
if hasattr(best_model, 'feature_importances_'):
    model_importance = pd.DataFrame({
        'feature': empath_cols,
        'model_importance': best_model.feature_importances_
    }).sort_values('model_importance', ascending=False)
    
    # Merge with SHAP importance
    comparison = feature_importance.merge(model_importance, on='feature')
    comparison = comparison.sort_values('mean_abs_shap', ascending=False).head(30)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    
    # SHAP importance
    axes[0].barh(range(len(comparison)), comparison['mean_abs_shap'].values, color='steelblue')
    axes[0].set_yticks(range(len(comparison)))
    axes[0].set_yticklabels(comparison['feature'].values)
    axes[0].set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold')
    axes[0].set_title('SHAP Importance', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Model importance
    axes[1].barh(range(len(comparison)), comparison['model_importance'].values, color='coral')
    axes[1].set_yticks(range(len(comparison)))
    axes[1].set_yticklabels(comparison['feature'].values)
    axes[1].set_xlabel('Model Feature Importance', fontsize=11, fontweight='bold')
    axes[1].set_title(f'{model_name} Importance', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('SHAP vs Model Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data-ai-slop-detector/shap_vs_model_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: data-ai-slop-detector/shap_vs_model_importance.png")
    plt.show()

# ===== SUMMARY STATISTICS =====
print("\n" + "="*80)
print("SHAP ANALYSIS SUMMARY")
print("="*80)
print(f"Model used: {model_name}")
print(f"Model accuracy: {max(rf_score, gb_score):.4f}")
print(f"Total features: {len(empath_cols)}")
print(f"Samples analyzed: {len(X_test)}")
print(f"\nMean |SHAP| statistics:")
print(f"  Mean: {mean_abs_shap.mean():.6f}")
print(f"  Median: {np.median(mean_abs_shap):.6f}")
print(f"  Std: {mean_abs_shap.std():.6f}")
print(f"  Min: {mean_abs_shap.min():.6f}")
print(f"  Max: {mean_abs_shap.max():.6f}")

# Identify key patterns
print(f"\n" + "="*80)
print("KEY INSIGHTS FROM TOP 10 FEATURES")
print("="*80)

for idx, row in feature_importance.head(10).iterrows():
    feature_idx = empath_cols.index(row['feature'])
    feature_shap = shap_values[:, feature_idx]
    feature_values = X_test[:, feature_idx]
    
    # Calculate correlation between feature value and SHAP value
    correlation = np.corrcoef(feature_values, feature_shap)[0, 1]
    
    direction = "HIGH values push toward BOT" if correlation > 0 else "LOW values push toward BOT"
    print(f"\n{row['feature']:25s} | SHAP = {row['mean_abs_shap']:.6f}")
    print(f"  → {direction} (correlation: {correlation:.3f})")

print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE")
print("="*80)