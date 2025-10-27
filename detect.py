import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This is the Hugging Face model ID for the short-text detector
# that matches the files in your screenshot.
MODEL_ID = "yuchuantian/AIGC_detector_env3"

# 1. Load the tokenizer and model
print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
print("Model loaded successfully.")

# 2. Define your text samples
text_to_check = [
    "This is a great post, I totally agree with everything you've said.", # Example 1
    "As a large language model, I am unable to provide personal opinions, but I can offer factual summaries.", # Example 2
    "Just popping in to say this is an incredibly nuanced take. Well done." # Example 3
]

# 3. Run detection on the text samples
print("-" * 30)
for text in text_to_check:
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Get model prediction (no need to calculate gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and calculate probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    # The model's config.json shows: "0": "human", "1": "AI"
    ai_prob = probabilities[0][1].item()
    prediction_index = torch.argmax(probabilities, dim=1).item()
    prediction_label = model.config.id2label[prediction_index] # Gets "human" or "AI"

    # 4. Print the result
    print(f"Text: \"{text}\"")
    print(f"Prediction: {prediction_label.upper()}")
    print(f"AI-Generated Confidence: {ai_prob:.2%}")
    print("-" * 30)