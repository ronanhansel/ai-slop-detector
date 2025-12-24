import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_recall_curve, f1_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Define confidence thresholds to test
CONFIDENCE_THRESHOLDS = [0.1, 0.3, 0.05, 0.01]
# Define minimum comment counts to test
MIN_COMMENT_COUNTS = [3, 5, 10]

print("Loading data...")
# Load the two datasets
df_classifier = pd.read_pickle('data-ai-slop-detector/classifier_data_prepared.pkl')
df_processed = pd.read_pickle('data-ai-slop-detector/final_detection_processed.pkl')

print(f"Classifier data shape: {df_classifier.shape}")
print(f"Processed data shape: {df_processed.shape}")

# Merge the datasets on a common key
if 'comment_id' in df_classifier.columns and 'comment_id' in df_processed.columns:
    df = pd.merge(df_classifier, df_processed, on='comment_id', suffixes=('', '_proc'))
else:
    df = pd.concat([df_classifier, df_processed], axis=1)

# Remove duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

print(f"Merged data shape: {df.shape}")

# Identify user column
if 'commenter_id' not in df.columns:
    print("Error: 'commenter_id' column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Define feature columns (ONLY for training - not labels)
avg_features = [
    'num_emojis', 'num_text_emojis', 'num_caps_words', 'num_unicode_chars',
    'contains_media', 'contains_link', 'num_tagged_people', 'tagged_grok',
    'used_slang', 'internet', 'swearing_terms', 'positive_emotion',
    'negative_emotion', 'technology', 'speaking', 'celebration',
    'dist_to_centroid_0', 'dist_to_centroid_1'
]

embedding_cols = ['empath_embedding', 'tf_idf_embedding', 'minilm_embedding']

# These are for target variable ONLY (not for training)
label_prob_pairs = [
    ('sentiment_label', 'sentiment_prob'),
    ('irony_label', 'irony_prob'),
    ('hate_label', 'hate_prob'),
    ('offensive_label', 'offensive_prob'),
    ('label', 'ai_confidence')
]

# Store results for all combinations
all_results = []

# ============================================================================
# TRAIN MODELS FOR EACH COMBINATION OF CONFIDENCE THRESHOLD AND MIN COMMENTS
# ============================================================================

for min_comments in MIN_COMMENT_COUNTS:
    print("\n" + "="*80)
    print(f"TRAINING MODELS FOR USERS WITH >= {min_comments} COMMENTS")
    print("="*80)
    
    for error_threshold in CONFIDENCE_THRESHOLDS:
        print("\n" + "-"*80)
        print(f"CONFIDENCE THRESHOLD: {error_threshold} | MIN COMMENTS: {min_comments}")
        print("-"*80)
        
        # Filter by confidence threshold: keep if confidence <= error_threshold OR >= (1 - error_threshold)
        df_conf_filtered = df[
            (df['ai_confidence'] <= error_threshold) | 
            (df['ai_confidence'] >= (1 - error_threshold))
        ].copy()
        
        print(f"Original data: {len(df)}")
        print(f"After confidence filter ({error_threshold}): {len(df_conf_filtered)}")
        print(f"  - Confidence <= {error_threshold}: {(df_conf_filtered['ai_confidence'] <= error_threshold).sum()}")
        print(f"  - Confidence >= {1 - error_threshold}: {(df_conf_filtered['ai_confidence'] >= (1 - error_threshold)).sum()}")
        
        # Filter users with at least min_comments comments
        user_counts = df_conf_filtered['commenter_id'].value_counts()
        valid_users = user_counts[user_counts >= min_comments].index
        df_filtered = df_conf_filtered[df_conf_filtered['commenter_id'].isin(valid_users)].copy()

        print(f"Users with >= {min_comments} comments: {len(valid_users)}")
        print(f"Filtered data shape: {df_filtered.shape}")
        
        if len(df_filtered) == 0:
            print(f"WARNING: No data available after filtering with threshold {error_threshold} and min_comments {min_comments}")
            continue

        # Initialize user-level features dictionary
        user_features = {}

        # Group by commenter_id
        grouped = df_filtered.groupby('commenter_id')

        # 1. Average features (convert bool to 0/1) - FOR TRAINING
        for col in avg_features:
            if col in df_filtered.columns:
                if df_filtered[col].dtype == bool:
                    user_features[f'avg_{col}'] = grouped[col].apply(lambda x: x.astype(int).mean())
                else:
                    user_features[f'avg_{col}'] = grouped[col].mean()

        # 2. Embedding features (mean and std, then reduce to single value)
        for emb_col in embedding_cols:
            if emb_col in df_filtered.columns:
                user_features[f'{emb_col}_mean_reduced'] = grouped[emb_col].apply(
                    lambda x: np.mean(np.vstack(x.values), axis=0).mean()
                )
                user_features[f'{emb_col}_std_reduced'] = grouped[emb_col].apply(
                    lambda x: np.std(np.vstack(x.values), axis=0).mean()
                )

        # 3. Weighted voting for labels (FOR TARGET VARIABLE ONLY)
        for label_col, prob_col in label_prob_pairs:
            if label_col in df_filtered.columns and prob_col in df_filtered.columns:
                def weighted_vote(group):
                    labels = group[label_col].values
                    probs = group[prob_col].values
                    unique_labels = np.unique(labels)
                    
                    weighted_sums = {}
                    for lbl in unique_labels:
                        mask = labels == lbl
                        weighted_sums[lbl] = probs[mask].sum()
                    
                    return max(weighted_sums, key=weighted_sums.get)
                
                user_features[f'{label_col}_weighted'] = grouped.apply(weighted_vote)

        # 4. Number of comments per user
        user_features['num_comments'] = grouped.size()

        # Create user-level dataframe
        user_df = pd.DataFrame(user_features)
        user_df = user_df.reset_index()

        print(f"\nUser-level dataframe shape: {user_df.shape}")

        # Save user-level features for this combination
        output_path = f'data-ai-slop-detector/user_level_features_classifier_min{min_comments}_thresh_{error_threshold}.pkl'
        user_df.to_pickle(output_path)
        print(f"Saved user-level features to: {output_path}")

        # Prepare ONLY training features
        training_features = [col for col in user_df.columns 
                            if (col.startswith('avg_') or col.endswith('_reduced') or col == 'num_comments')
                            and not col.endswith('_weighted')]

        print(f"\nTraining features: {len(training_features)}")

        X = user_df[training_features].values

        # Target variable
        if 'label_weighted' not in user_df.columns:
            print("Error: 'label_weighted' column not found!")
            continue

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(user_df['label_weighted'])

        print(f"\nClass distribution:")
        for i, class_name in enumerate(le.classes_):
            count = (y == i).sum()
            pct = count / len(y) * 100
            print(f"  {class_name}: {count} ({pct:.2f}%)")

        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # XGBoost training
        print(f"\nTraining XGBoost model...")
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }

        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=training_features)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=training_features)

        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        error_rate = 1 - accuracy

        print("\n" + "*"*80)
        print(f"TEST SET RESULTS (Min Comments={min_comments}, Threshold={error_threshold})")
        print("*"*80)
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Error Rate:    {error_rate:.4f}")
        print(f"ROC-AUC:       {roc_auc:.4f}")
        print(f"F1-Score:      {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Cross-validation scores
        print("\n5-fold cross-validation...")
        cv_model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != 'eval_metric'})
        cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=le.classes_, yticklabels=le.classes_)
        axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Training history
        axes[1, 0].plot(evals_result['train']['auc'], label='Train AUC')
        axes[1, 0].plot(evals_result['test']['auc'], label='Test AUC')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Training History', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 1].plot(recall, precision, linewidth=2, label='PR Curve')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'data-ai-slop-detector/user_level_model_results_min{min_comments}_thresh_{error_threshold}.png', 
                    dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: user_level_model_results_min{min_comments}_thresh_{error_threshold}.png")

        # Feature importance (top 20)
        importance_dict = model.get_score(importance_type='weight')
        if importance_dict:
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance (Weight)')
            plt.title(f'Top 20 Feature Importance (Min Comments={min_comments}, Threshold={error_threshold})', fontsize=12, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'data-ai-slop-detector/user_level_feature_importance_min{min_comments}_thresh_{error_threshold}.png', 
                        dpi=150, bbox_inches='tight')
            print(f"Feature importance saved: user_level_feature_importance_min{min_comments}_thresh_{error_threshold}.png")

        # Save model and scaler
        model.save_model(f'data-ai-slop-detector/user_level_xgboost_model_min{min_comments}_thresh_{error_threshold}.json')
        
        import pickle
        with open(f'data-ai-slop-detector/user_level_scaler_min{min_comments}_thresh_{error_threshold}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'data-ai-slop-detector/user_level_label_encoder_min{min_comments}_thresh_{error_threshold}.pkl', 'wb') as f:
            pickle.dump(le, f)

        # Save summary metrics
        summary = {
            'min_comments': min_comments,
            'confidence_threshold': error_threshold,
            'num_comments_after_filter': len(df_conf_filtered),
            'num_users': len(user_df),
            'num_features': X.shape[1],
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'error_rate': error_rate,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
        
        all_results.append(summary)

        # Save CSV version of user features
        user_df[training_features + ['commenter_id', 'label_weighted']].to_csv(
            f'data-ai-slop-detector/user_level_features_classifier_min{min_comments}_thresh_{error_threshold}.csv', 
            index=False)
        print(f"Saved user features CSV: user_level_features_classifier_min{min_comments}_thresh_{error_threshold}.csv")

# ============================================================================
# SUMMARY COMPARISON OF ALL COMBINATIONS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: COMPARISON ACROSS ALL MIN COMMENT COUNTS AND CONFIDENCE THRESHOLDS")
print("="*80)

summary_df = pd.DataFrame(all_results)
print("\n" + summary_df.to_string(index=False))

# Save comparison
summary_df.to_csv('data-ai-slop-detector/user_level_model_summary_all_combinations.csv', index=False)
print("\nSummary comparison saved to: user_level_model_summary_all_combinations.csv")

# Separate reports for each min_comments value
for min_comments in MIN_COMMENT_COUNTS:
    print("\n" + "="*80)
    print(f"DETAILED REPORT FOR MIN COMMENTS = {min_comments}")
    print("="*80)
    
    subset_results = summary_df[summary_df['min_comments'] == min_comments]
    print("\n" + subset_results.to_string(index=False))
    
    # Save subset report
    subset_results.to_csv(f'data-ai-slop-detector/user_level_model_summary_min{min_comments}.csv', index=False)
    print(f"\nSaved report to: user_level_model_summary_min{min_comments}.csv")
    
    # Visualize comparison for this min_comments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy comparison
    axes[0, 0].plot(subset_results['confidence_threshold'], subset_results['accuracy'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Accuracy vs Confidence Threshold (Min Comments={min_comments})', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # ROC-AUC comparison
    axes[0, 1].plot(subset_results['confidence_threshold'], subset_results['roc_auc'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title(f'ROC-AUC vs Confidence Threshold (Min Comments={min_comments})', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # F1 comparison
    axes[1, 0].plot(subset_results['confidence_threshold'], subset_results['f1_score'], 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title(f'F1-Score vs Confidence Threshold (Min Comments={min_comments})', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Data size comparison
    axes[1, 1].plot(subset_results['confidence_threshold'], subset_results['num_comments_after_filter'], 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Confidence Threshold')
    axes[1, 1].set_ylabel('Number of Comments')
    axes[1, 1].set_title(f'Data Size vs Confidence Threshold (Min Comments={min_comments})', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data-ai-slop-detector/user_level_model_comparison_min{min_comments}.png', dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved: user_level_model_comparison_min{min_comments}.png")

# ============================================================================
# OVERALL COMPARISON ACROSS ALL MIN COMMENT COUNTS
# ============================================================================

print("\n" + "="*80)
print("OVERALL COMPARISON ACROSS ALL MIN COMMENT COUNTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for min_comments in MIN_COMMENT_COUNTS:
    subset = summary_df[summary_df['min_comments'] == min_comments]
    
    axes[0, 0].plot(subset['confidence_threshold'], subset['accuracy'], 'o-', label=f'Min Comments={min_comments}', linewidth=2, markersize=8)
    axes[0, 1].plot(subset['confidence_threshold'], subset['roc_auc'], 'o-', label=f'Min Comments={min_comments}', linewidth=2, markersize=8)
    axes[1, 0].plot(subset['confidence_threshold'], subset['f1_score'], 'o-', label=f'Min Comments={min_comments}', linewidth=2, markersize=8)
    axes[1, 1].plot(subset['confidence_threshold'], subset['num_users'], 'o-', label=f'Min Comments={min_comments}', linewidth=2, markersize=8)

axes[0, 0].set_xlabel('Confidence Threshold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_xlabel('Confidence Threshold')
axes[0, 1].set_ylabel('ROC-AUC')
axes[0, 1].set_title('ROC-AUC Comparison', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_xlabel('Confidence Threshold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_xlabel('Confidence Threshold')
axes[1, 1].set_ylabel('Number of Users')
axes[1, 1].set_title('User Count Comparison', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data-ai-slop-detector/user_level_model_comparison_all_min_comments.png', dpi=150, bbox_inches='tight')
print("Overall comparison visualization saved: user_level_model_comparison_all_min_comments.png")

print("\n" + "="*80)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("="*80)