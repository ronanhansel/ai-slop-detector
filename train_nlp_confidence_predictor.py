import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading data...")
# Load the dataset
df = pd.read_pickle('data-ai-slop-detector/final_detection_processed.pkl')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check for required columns
required_cols = ['minilm_embedding', 'ai_confidence']
for col in required_cols:
    if col not in df.columns:
        print(f"Error: '{col}' not found in data!")
        exit(1)

# Remove rows with missing values
df_clean = df.dropna(subset=required_cols).copy()
print(f"\nAfter removing NaN: {len(df_clean)} rows")

# Extract embeddings
X = np.vstack(df_clean['minilm_embedding'].values)
y = df_clean['ai_confidence'].values

print(f"\nEmbedding shape: {X.shape}")
print(f"Confidence shape: {y.shape}")
print(f"Confidence stats - Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

# Standardize embeddings
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled).to(device)
X_test_tensor = torch.from_numpy(X_test_scaled).to(device)
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1).to(device)

# ============================================================================
# BUILD NEURAL NETWORK MODEL
# ============================================================================

print("\n" + "="*70)
print("BUILDING NEURAL NETWORK MODEL")
print("="*70)

embedding_dim = X_train_scaled.shape[1]
print(f"Embedding dimension: {embedding_dim}")

class ConfidencePredictorNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(ConfidencePredictorNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate - 0.1)
        
        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(dropout_rate - 0.1)
        
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
        # L2 regularization
        self.l2_lambda = 1e-4
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        x = self.sigmoid(x)
        
        return x
    
    def l2_penalty(self):
        """Calculate L2 regularization penalty"""
        l2 = 0
        for param in self.parameters():
            l2 += torch.norm(param)
        return self.l2_lambda * l2

# Create model
model = ConfidencePredictorNet(embedding_dim).to(device)
print(f"\nModel created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

batch_size = 32
epochs = 200
patience = 15
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation split (20% of training)
val_size = int(0.2 * len(X_train_tensor))
train_size = len(X_train_tensor) - val_size
train_indices = torch.randperm(len(X_train_tensor))[:train_size]
val_indices = torch.randperm(len(X_train_tensor))[train_size:]

X_train_split = X_train_tensor[train_indices]
y_train_split = y_train_tensor[train_indices]
X_val = X_train_tensor[val_indices]
y_val = y_train_tensor[val_indices]

train_split_dataset = TensorDataset(X_train_split, y_train_split)
train_split_loader = DataLoader(train_split_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_split_loader:
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Add L2 regularization
        loss += model.l2_penalty()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_split_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)
        val_loss += model.l2_penalty()
        val_loss = val_loss.item()
    
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'data-ai-slop-detector/best_model.pt')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('data-ai-slop-detector/best_model.pt'))
print(f"\nTraining completed. Epochs trained: {epoch+1}")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70)

model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy().flatten()

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Metrics:")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  R²:   {r2:.6f}")

# For binary classification (confidence > 0.5 = positive class)
y_pred_binary = (y_pred >= 0.5).astype(int)
y_test_binary = (y_test >= 0.5).astype(int)

accuracy = accuracy_score(y_test_binary, y_pred_binary)
auc = roc_auc_score(y_test_binary, y_pred)

print(f"\nClassification Metrics (threshold=0.5):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC:      {auc:.4f}")

cm = confusion_matrix(y_test_binary, y_pred_binary)
print(f"\nConfusion Matrix:")
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=['Low Confidence', 'High Confidence']))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Training history - Loss
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Model Loss Over Epochs', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Predicted vs Actual confidence scores
axes[0, 1].scatter(y_test, y_pred, alpha=0.5, s=20)
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Confidence')
axes[0, 1].set_ylabel('Predicted Confidence')
axes[0, 1].set_title('Predicted vs Actual Confidence Scores', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim([-0.05, 1.05])
axes[0, 1].set_ylim([-0.05, 1.05])

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
axes[0, 2].set_title('Confusion Matrix (threshold=0.5)', fontweight='bold')
axes[0, 2].set_ylabel('True Label')
axes[0, 2].set_xlabel('Predicted Label')

# Residuals distribution
residuals = y_test - y_pred
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Distribution of predicted vs actual
axes[1, 1].hist(y_test, bins=50, alpha=0.6, label='Actual', edgecolor='black')
axes[1, 1].hist(y_pred, bins=50, alpha=0.6, label='Predicted', edgecolor='black')
axes[1, 1].set_xlabel('Confidence Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Confidence Scores', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Error distribution by actual confidence level
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
errors_by_bin = []
for i in range(len(bins)-1):
    mask = (y_test >= bins[i]) & (y_test < bins[i+1])
    if mask.sum() > 0:
        errors_by_bin.append(np.abs(y_test[mask] - y_pred[mask]).mean())
    else:
        errors_by_bin.append(np.nan)

axes[1, 2].bar(bin_centers, errors_by_bin, width=0.08, edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Actual Confidence Score')
axes[1, 2].set_ylabel('Mean Absolute Error')
axes[1, 2].set_title('Prediction Error by Confidence Level', fontweight='bold')
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data-ai-slop-detector/nlp_confidence_predictor_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved: nlp_confidence_predictor_results.png")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING ARTIFACTS")
print("="*70)

# Save model
torch.save(model.state_dict(), 'data-ai-slop-detector/nlp_confidence_predictor_model.pt')
print("Model saved: nlp_confidence_predictor_model.pt")

# Save scaler
import pickle
with open('data-ai-slop-detector/nlp_confidence_predictor_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved: nlp_confidence_predictor_scaler.pkl")

# Save summary metrics
summary = {
    'embedding_dimension': embedding_dim,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'epochs_trained': epoch + 1,
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2_score': r2,
    'accuracy': accuracy,
    'auc': auc,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'device': str(device),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('data-ai-slop-detector/nlp_confidence_predictor_summary.csv', index=False)
print("Summary saved: nlp_confidence_predictor_summary.csv")

# Save predictions for inspection
predictions_df = pd.DataFrame({
    'actual_confidence': y_test,
    'predicted_confidence': y_pred,
    'prediction_error': np.abs(y_test - y_pred),
    'actual_binary': y_test_binary,
    'predicted_binary': y_pred_binary
})
predictions_df.to_csv('data-ai-slop-detector/nlp_confidence_predictor_predictions.csv', index=False)
print("Predictions saved: nlp_confidence_predictor_predictions.csv")

print("\n" + "="*70)
print("SUMMARY RESULTS")
print("="*70)
print(f"\nRegression Performance:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  R²:   {r2:.6f}")
print(f"\nClassification Performance (threshold=0.5):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC:      {auc:.4f}")
print(f"\nModel Info:")
print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Epochs Trained: {epoch + 1}")
print(f"  Device: {device}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)