import os
import random
import string
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate  # Hugging Face's evaluate library

# --- 1. Configuration ---

# --- Model and Dataset ---
MODEL_NAME = "microsoft/deberta-v3-small"
DATASET_NAME = "gsingh1-py/train"
DATASET_SPLIT = "train"

# --- Labels for Task-B ---
LABEL_COLUMNS = [
    "Human_story",
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o"
]


NUM_LABELS = len(LABEL_COLUMNS)

# Create label mappings
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_COLUMNS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_COLUMNS)}

# --- Output Directories ---
MODEL_1_OUTPUT_DIR = "./deberta-v3-small-noisy-only"
MODEL_2A_OUTPUT_DIR = "./deberta-v3-small-original"
MODEL_2B_OUTPUT_DIR = "./deberta-v3-small-double-finetune"


# --- 2. Data "Noising" Function (as per paper) ---

def add_noise_to_example(example, noise_ratio=0.1):
    """
    Injects 10% junk words into the text, as described in
    Section 4.2.
    """
    # The transformation step will create this 'Text' field
    text = example['Text']
    words = text.split()
    
    if not words:
        return example

    num_to_inject = int(len(words) * noise_ratio)
    inject_indices = sorted(random.sample(range(len(words) + 1), num_to_inject))
    
    new_words = []
    word_idx = 0
    for inject_idx in inject_indices:
        new_words.extend(words[word_idx:inject_idx])
        
        # "randomly generated with lengths varying between 3 to 8 characters"
        junk_len = random.randint(3, 8)
        junk_word = ''.join(random.choices(string.ascii_lowercase, k=junk_len))
        new_words.append(junk_word)
        
        word_idx = inject_idx
        
    new_words.extend(words[word_idx:])
    example['Text'] = ' '.join(new_words)
    return example

# --- 3. Data Loading and Transformation ---

def transform_to_long_format(dataset):
    """
    Transforms the 'wide' dataset (gsingh1-py/train) into the 'long'
    format (Text, Label_B) described in the paper.
    """
    for row in dataset:
        for col_name in LABEL_COLUMNS:
            if row[col_name]:  # Ensure text is not null
                yield {
                    "Text": row[col_name],
                    "Label_B": col_name
                }

# *** THIS IS THE UPDATED SECTION ***
print(f"Loading raw dataset '{DATASET_NAME}' (split='{DATASET_SPLIT}') from the Hugging Face Hub...")
raw_ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
# **********************************

# --- Sanity check for column names ---
print("Checking for required columns...")
missing_cols = [col for col in LABEL_COLUMNS if col not in raw_ds.column_names]
if missing_cols:
    print(f"\n[ERROR] The following columns are missing from the dataset:")
    for col in missing_cols:
        print(f"- {col}")
    print("\nPlease check the `LABEL_COLUMNS` and `LABEL_MAP` variables.")
    print("Dataset column names are case-sensitive and might replace '/' with '_'.")
    print("Actual columns found:", raw_ds.column_names)
    exit()
else:
    print("All required columns found.")
# -------------------------------------


print("Transforming dataset to long format ('Text', 'Label_B')...")
# Create the long-format dataset
long_format_ds = Dataset.from_generator(
    transform_to_long_format,
    gen_kwargs={"dataset": raw_ds}
)

# --- 4. Create Splits and "Noisy" Dataset ---

# The paper mentions a validation set. We'll create one.
print("Splitting into train and validation sets...")
# Shuffle and split. Using a 10% test split for validation.
train_val_split = long_format_ds.shuffle(seed=42).train_test_split(
    test_size=0.1
)

original_train_dataset = train_val_split['train']
validation_dataset = train_val_split['test']

print(f"Original train set size: {len(original_train_dataset)}")
print(f"Validation set size:   {len(validation_dataset)}")

print("Generating 'Noisy' training dataset (injecting 10% junk words)...")
# "we noised our dataset by injecting 10% junk or garbled words"
noisy_train_dataset = original_train_dataset.map(
    add_noise_to_example, 
    num_proc=os.cpu_count()
)

print("\n--- Example of Noised Data ---")
original_sample = original_train_dataset[0]
noisy_sample = noisy_train_dataset[0]
print(f"Original: {original_sample['Text'][:100]}...")
print(f"Label:    {original_sample['Label_B']}")
print(f"Noised:   {noisy_sample['Text'][:100]}...")
print(f"Label:    {noisy_sample['Label_B']}")
print("-" * 30)

# --- 5. Tokenization ---

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    max_length=768, # "max token length: 768"
    use_fast=False # DeBERTa-v3 fast tokenizer depends on tiktoken assets that are not bundled
)

def preprocess_function(examples):
    """Tokenize text and convert labels to IDs."""
    tokenized = tokenizer(
        examples['Text'],
        truncation=True,
        padding=False,  # Trainer will handle padding
        max_length=768
    )
    tokenized['label'] = [LABEL_TO_ID[label] for label in examples['Label_B']]
    return tokenized

print("Tokenizing all datasets...")
tokenized_original_train = original_train_dataset.map(
    preprocess_function, batched=True, num_proc=os.cpu_count()
)
tokenized_noisy_train = noisy_train_dataset.map(
    preprocess_function, batched=True, num_proc=os.cpu_count()
)
tokenized_validation = validation_dataset.map(
    preprocess_function, batched=True, num_proc=os.cpu_count()
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 6. Metrics and Training Arguments ---

# Define metrics
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(
        predictions=predictions, 
        references=labels, 
        average="macro"
    )

def get_training_args(output_dir):
    """
    Returns TrainingArguments based on Table 5 for Label-B
    """
    return TrainingArguments(
        output_dir=output_dir,
        # --- Key Hyperparameters from Table 5 ---
        learning_rate=5e-05,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=24,
        num_train_epochs=5,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        warmup_steps=500,
        # --- Other standard args ---
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

def get_model():
    """Loads a fresh DeBERTa-v3-small for classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )

# --- 7. Run 1: Finetune on Noisy Data (Model 1) ---
print(f"\n--- Starting Run 1: Finetuning on Noisy Data (Model 1) ---")
print(f"Output directory: {MODEL_1_OUTPUT_DIR}")

model_1 = get_model()
training_args_1 = get_training_args(MODEL_1_OUTPUT_DIR)

trainer_1 = Trainer(
    model=model_1,
    args=training_args_1,
    train_dataset=tokenized_noisy_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training Model 1...")
trainer_1.train()
print("Saving Model 1...")
trainer_1.save_model(MODEL_1_OUTPUT_DIR)
print("--- Model 1 Training Complete ---")


# --- 8. Run 2: Double Finetune (Model 2) ---
print(f"\n--- Starting Run 2: Double Finetuning (Model 2) ---")

# --- Stage 2a: Train on Original Data ---
print(f"Stage 2a: Training on Original Data...")
print(f"Output directory: {MODEL_2A_OUTPUT_DIR}")

model_2a = get_model()
training_args_2a = get_training_args(MODEL_2A_OUTPUT_DIR)

trainer_2a = Trainer(
    model=model_2a,
    args=training_args_2a,
    train_dataset=tokenized_original_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training Model 2a...")
trainer_2a.train()
print("Saving Model 2a...")
trainer_2a.save_model(MODEL_2A_OUTPUT_DIR)
print("--- Model 2a Training Complete ---")


# --- Stage 2b: Further Finetune on Noisy Data ---
print(f"\nStage 2b: Loading Model 2a and finetuning on Noisy Data...")
print(f"Output directory: {MODEL_2B_OUTPUT_DIR}")

# Load the model we just trained
model_2b = AutoModelForSequenceClassification.from_pretrained(
    MODEL_2A_OUTPUT_DIR, # <-- Load the saved model
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)

training_args_2b = get_training_args(MODEL_2B_OUTPUT_DIR)

trainer_2b = Trainer(
    model=model_2b,
    args=training_args_2b,
    train_dataset=tokenized_noisy_train, # Train on noisy data
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training Model 2b...")
trainer_2b.train()
print("Saving Model 2b (final Model 2)...")
trainer_2b.save_model(MODEL_2B_OUTPUT_DIR)
print("--- Model 2b Training Complete ---")


# --- 9. Ensemble Prediction Logic (Inference) ---
print("\n" + "="*50)
print("      Ensemble Prediction Logic (Inference)")
print("="*50)

# Load the two final models
try:
    ensemble_model_1 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_1_OUTPUT_DIR
    )
    ensemble_model_2 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_2B_OUTPUT_DIR
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model_1.to(device)
    ensemble_model_2.to(device)
    ensemble_model_1.eval()
    ensemble_model_2.eval()
    
    MODELS_LOADED = True
    print("Successfully loaded finetuned models for ensemble.")

except Exception as e:
    print(f"Could not load finetuned models. Using base models as placeholders.")
    print(f"Error: {e}")
    ensemble_model_1 = get_model()
    ensemble_model_2 = get_model()
    MODELS_LOADED = False


def predict_ensemble(text):
    """
    Combines Model 1 and Model 2 with a 60:40 weighted average
    as described in the paper.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits_1 = ensemble_model_1(**inputs).logits
        logits_2 = ensemble_model_2(**inputs).logits
        
    # Apply softmax to get probabilities
    probs_1 = torch.softmax(logits_1, dim=-1)
    probs_2 = torch.softmax(logits_2, dim=-1)
    
    # "a weighted (60:40) ensemble model for Task-B"
    # 60% weight to Model 1 (noisy-only)
    # 40% weight to Model 2 (double-finetune)
    final_probs = (0.6 * probs_1) + (0.4 * probs_2)
    
    # Get the final prediction
    prediction_id = torch.argmax(final_probs, dim=-1).item()
    return ID_TO_LABEL[prediction_id]

# --- Example Prediction ---
test_text = validation_dataset[0]['Text']
true_label = validation_dataset[0]['Label_B']
prediction = predict_ensemble(test_text)

print("\n--- Example Ensemble Prediction ---")
print(f"Text:         '{test_text[:150]}...'")
print(f"True Label:   {true_label}")
print(f"Ensemble Prediction: {prediction}")
if not MODELS_LOADED:
    print("\n[Warning] Prediction is from *untrained* models.")