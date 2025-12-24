import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_recall_curve, f1_score,
    precision_score, recall_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING CLASSIFIER WITH ALL FEATURES")
print("="*80)

# Check if prepared data exists, otherwise prepare it
data_path = Path('data-ai-slop-detector/classifier_data_prepared.pkl')
if not data_path.exists():
    print("\nPrepared data not found. Running data preparation script...")
    import subprocess
    subprocess.run(['python', 'prepare_classifier_data.py'])
    print("\nData preparation complete. Loading prepared data...")

# Load prepared data
print("\nLoading prepared data...")
with open('data-ai-slop-detector/classifier_data_prepared.pkl', 'rb') as f:
    df = pd.read_pickle(f)

print(f"Loaded data shape: {df.shape}")

# Define feature columns
behavioral_features = [
    'num_emojis', 'num_text_emojis', 'num_caps_words', 'num_unicode_chars',
    'contains_media', 'contains_link', 'num_tagged_people', 'tagged_grok', 'used_slang'
]

empath_features = [
    'internet', 'swearing_terms', 'positive_emotion', 'negative_emotion',
    'technology', 'speaking', 'celebration'
]

distance_features = ['dist_to_centroid_0', 'dist_to_centroid_1']

all_features = behavioral_features + empath_features + distance_features

print(f"\nFeature groups:")
print(f"  Behavioral: {len(behavioral_features)}")
print(f"  Empath: {len(empath_features)}")
print(f"  MiniLM distances: {len(distance_features)}")
print(f"  Total: {len(all_features)}")

# Function to prepare dataset at given confidence threshold
def prepare_confidence_dataset(df, threshold_low, threshold_high, name):
    """Filter dataset by confidence thresholds"""
    df_filtered = df[
        (df['ai_confidence'] >= threshold_high) | 
        (df['ai_confidence'] <= threshold_low)
    ].copy()
    
    print(f"\n{name}:")
    print(f"  Original size: {len(df)}")
    print(f"  Filtered size: {len(df_filtered)}")
    print(f"  Percentage kept: {len(df_filtered)/len(df)*100:.2f}%")
    
    return df_filtered

# Create three confidence-based datasets
datasets = {
    'Full_Data': df.copy(),
    'Conf_0.7': prepare_confidence_dataset(df, 0.3, 0.7, "Confidence ≥0.7 or ≤0.3"),
    'Conf_0.85': prepare_confidence_dataset(df, 0.15, 0.85, "Confidence ≥0.85 or ≤0.15"),
    'Conf_0.9': prepare_confidence_dataset(df, 0.1, 0.9, "Confidence ≥0.9 or ≤0.1"),
    'Conf_0.95': prepare_confidence_dataset(df, 0.05, 0.95, "Confidence ≥0.95 or ≤0.05"),
    'Conf_0.975': prepare_confidence_dataset(df, 0.025, 0.975, "Confidence ≥0.975 or ≤0.025"),
    'Conf_0.99': prepare_confidence_dataset(df, 0.01, 0.99, "Confidence ≥0.99 or ≤0.01"),
}


# Function to train and evaluate model
def train_and_evaluate(data, dataset_name, features):
    """Train XGBoost classifier and evaluate performance"""
    
    print("\n" + "="*80)
    print(f"TRAINING MODEL: {dataset_name}")
    print("="*80)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data['label'])
    X = data[features].copy()
    
    print(f"\nClass distribution:")
    for i, label in enumerate(le.classes_):
        print(f"  {label}: {(y == i).sum()} ({(y == i).sum()/len(y)*100:.2f}%)")
    
    # Train-val-test split (60-20-20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost training
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=features)
    
    print("\nTraining XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    
    # Find best threshold on validation set
    y_val_proba = model.predict(dval)
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    
    print("\nFinding optimal threshold on validation set...")
    for thresh in thresholds:
        y_val_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"Best threshold: {best_threshold:.3f} (Validation F1: {best_f1:.4f})")
    
    # Predictions on test set with optimized threshold
    y_test_proba = model.predict(dtest)
    y_pred = (y_test_proba >= best_threshold).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n" + "-"*50)
    print("TEST SET RESULTS (Optimized Threshold)")
    print("-"*50)
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Feature importance
    importance_dict = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # === REPORT 1: AUC CURVES ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes1[0].plot(fpr, tpr, linewidth=3, label=f'AUC = {roc_auc:.4f}', color='#2ecc71')
    axes1[0].plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', alpha=0.5, linewidth=2)
    axes1[0].fill_between(fpr, tpr, alpha=0.2, color='#2ecc71')
    axes1[0].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    axes1[0].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    axes1[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes1[0].legend(fontsize=12, loc='lower right')
    axes1[0].grid(alpha=0.3, linestyle='--')
    axes1[0].set_xlim([0, 1])
    axes1[0].set_ylim([0, 1])
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    from sklearn.metrics import auc as compute_auc
    pr_auc = compute_auc(recall_curve, precision_curve)
    axes1[1].plot(recall_curve, precision_curve, linewidth=3, label=f'PR AUC = {pr_auc:.4f}', color='#3498db')
    axes1[1].fill_between(recall_curve, precision_curve, alpha=0.2, color='#3498db')
    baseline = (y_test == 1).sum() / len(y_test)
    axes1[1].axhline(y=baseline, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline = {baseline:.3f}')
    axes1[1].set_xlabel('Recall', fontsize=13, fontweight='bold')
    axes1[1].set_ylabel('Precision', fontsize=13, fontweight='bold')
    axes1[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes1[1].legend(fontsize=12, loc='best')
    axes1[1].grid(alpha=0.3, linestyle='--')
    axes1[1].set_xlim([0, 1])
    axes1[1].set_ylim([0, 1])
    
    fig1.suptitle(f'AUC Curves: {dataset_name}\n({len(data)} samples, Optimized Threshold: {best_threshold:.3f})',
                  fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file1 = f'data-ai-slop-detector/report_auc_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"\nSaved AUC report to: {output_file1}")
    plt.close()
    
    # === REPORT 2: CONFUSION MATRIX & METRICS ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16, 'weight': 'bold'})
    axes2[0].set_title(f'Confusion Matrix (Threshold: {best_threshold:.3f})', fontsize=14, fontweight='bold')
    axes2[0].set_ylabel('True Label', fontsize=13, fontweight='bold')
    axes2[0].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Add percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)'
            axes2[0].text(j + 0.5, i + 0.7, text, ha='center', va='center', 
                         fontsize=11, color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Performance Metrics
    metrics_data = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    colors_metrics = []
    for v in metrics_data.values():
        if v >= 0.8:
            colors_metrics.append('#2ecc71')  # green
        elif v >= 0.7:
            colors_metrics.append('#3498db')  # blue
        elif v >= 0.6:
            colors_metrics.append('#f39c12')  # orange
        else:
            colors_metrics.append('#e74c3c')  # red
    
    bars = axes2[1].bar(metrics_data.keys(), metrics_data.values(), 
                        color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=2)
    axes2[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (0.5)')
    axes2[1].axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.7)')
    axes2[1].axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (0.8)')
    axes2[1].set_ylabel('Score', fontsize=13, fontweight='bold')
    axes2[1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes2[1].set_ylim(0, 1.05)
    axes2[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes2[1].legend(fontsize=10, loc='lower right')
    axes2[1].set_xticklabels(metrics_data.keys(), fontsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_data.values()):
        height = bar.get_height()
        axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig2.suptitle(f'Confusion Matrix & Metrics: {dataset_name}\n({len(data)} samples, Test Set: {len(X_test)})',
                  fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file2 = f'data-ai-slop-detector/report_cm_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved Confusion Matrix report to: {output_file2}")
    plt.close()
    
    # Save model
    model_file = f'data-ai-slop-detector/xgb_model_{dataset_name.replace(" ", "_")}.json'
    model.save_model(model_file)
    print(f"Saved model to: {model_file}")
    
    # Save scaler and threshold
    metadata = {
        'scaler': scaler,
        'threshold': best_threshold,
        'label_encoder': le
    }
    metadata_file = f'data-ai-slop-detector/metadata_{dataset_name.replace(" ", "_")}.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {metadata_file}")
    
    return {
        'dataset': dataset_name,
        'n_samples': len(data),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'threshold': best_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance_df.to_dict('records'),
    }


# Train models on all datasets
results = []

for dataset_name, dataset in datasets.items():
    if len(dataset) < 100:
        print(f"\nSkipping {dataset_name}: insufficient data ({len(dataset)} samples)")
        continue
    
    result = train_and_evaluate(dataset, dataset_name, all_features)
    results.append(result)

# ===== GENERATE FINAL REPORT =====
print("\n" + "="*80)
print("GENERATING FINAL REPORT")
print("="*80)

# Create summary DataFrame
summary_df = pd.DataFrame([
    {
        'Dataset': r['dataset'],
        'Samples': r['n_samples'],
        'Train': r['n_train'],
        'Test': r['n_test'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1_score'],
        'ROC-AUC': r['roc_auc']
    }
    for r in results
])

print("\n" + "-"*80)
print("PERFORMANCE SUMMARY ACROSS ALL DATASETS")
print("-"*80)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('data-ai-slop-detector/classifier_performance_summary.csv', index=False)
print("\nSaved summary to: data-ai-slop-detector/classifier_performance_summary.csv")

# Create comparative visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(summary_df))
width = 0.15

for i, metric in enumerate(metrics):
    axes[0, 0].bar(x + i*width, summary_df[metric], width, label=metric, alpha=0.8)

axes[0, 0].set_xlabel('Dataset', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x + width * 2)
axes[0, 0].set_xticklabels(summary_df['Dataset'], rotation=15, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim(0, 1.1)

# 2. Sample Size Comparison
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = axes[0, 1].bar(summary_df['Dataset'], summary_df['Samples'], color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Dataset Size Comparison', fontsize=13, fontweight='bold')
axes[0, 1].set_xticklabels(summary_df['Dataset'], rotation=15, ha='right')
axes[0, 1].grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, summary_df['Samples']):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Feature Group Importance Comparison
group_data = []
for r in results:
    for group, importance in r['group_importance'].items():
        group_data.append({
            'Dataset': r['dataset'],
            'Group': group,
            'Importance': importance
        })

group_df = pd.DataFrame(group_data)
pivot_group = group_df.pivot(index='Group', columns='Dataset', values='Importance')

pivot_group.plot(kind='bar', ax=axes[1, 0], alpha=0.8, edgecolor='black')
axes[1, 0].set_ylabel('Total Importance', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Feature Group Importance by Dataset', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Feature Group', fontsize=11, fontweight='bold')
axes[1, 0].legend(title='Dataset', title_fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# 4. Heatmap of Metrics
metrics_heatmap = summary_df.set_index('Dataset')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
sns.heatmap(metrics_heatmap.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 1],
            vmin=0.5, vmax=1.0, cbar_kws={'label': 'Score'})
axes[1, 1].set_title('Performance Metrics Heatmap', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Dataset', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Metric', fontsize=11, fontweight='bold')

plt.suptitle('XGBoost Classifier: Final Performance Report\n' + 
             f'Features: {len(all_features)} (Behavioral: {len(behavioral_features)}, ' +
             f'Empath: {len(empath_features)}, MiniLM: {len(distance_features)})',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('data-ai-slop-detector/classifier_final_report.png', dpi=150, bbox_inches='tight')
print("\nSaved final report to: data-ai-slop-detector/classifier_final_report.png")
plt.show()

# Save full results
with open('data-ai-slop-detector/classifier_full_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved full results to: data-ai-slop-detector/classifier_full_results.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nTotal experiments: {len(results)}")
print(f"Best performing dataset: {summary_df.loc[summary_df['F1-Score'].idxmax(), 'Dataset']}")
print(f"Best F1-Score: {summary_df['F1-Score'].max():.4f}")
print(f"Best ROC-AUC: {summary_df['ROC-AUC'].max():.4f}")