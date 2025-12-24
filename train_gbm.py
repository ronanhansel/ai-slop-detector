import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_recall_curve, f1_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_pickle('data-ai-slop-detector/final_detection.pkl')

# Features to use
features = [
    'num_emojis',
    'num_text_emojis',
    'num_caps_words',
    'num_unicode_chars',
    'contains_media',
    'contains_link',
    'num_tagged_people',
    'tagged_grok',
    'used_slang',
]

err_threshold = 0.05
# Keep only rows with label, ai_confidence and required features, then filter by high/low ai_confidence
df_clean = df.dropna(subset=['label', 'ai_confidence'] + features).copy()
initial_len = len(df_clean)
df_clean = df_clean[(df_clean['ai_confidence'] >= (1 - err_threshold)) | (df_clean['ai_confidence'] <= err_threshold)].copy()
print(f"Dataset size after cleaning: {initial_len} -> after ai_confidence filter: {len(df_clean)}")

# Encode target (assuming 'label' is 'AI' or 'human')
le = LabelEncoder()
y = le.fit_transform(df_clean['label'])
X = df_clean[features].copy()

print(f"\nClass distribution:")
print(f"  {le.classes_[0]}: {(y == 0).sum()}")
print(f"  {le.classes_[1]}: {(y == 1).sum()}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

# XGBoost training
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 7,
    'random_state': 42,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False,
)

# Predictions
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluation
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Feature importance
importance_dict = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Feature Importance')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1, 1].plot(recall, precision, label='PR Curve')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data-ai-slop-detector/xgb_results.png', dpi=150)
print("\nVisualization saved to 'data-ai-slop-detector/xgb_results.png'")

# Save model
model.save_model('data-ai-slop-detector/xgboost_model.json')
print("Model saved to 'data-ai-slop-detector/xgboost_model.json'")