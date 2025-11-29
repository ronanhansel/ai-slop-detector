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

def detect(text):
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

    return {
        "text": text,
        "prediction_label": prediction_label,
        "ai_prob": ai_prob
    }

if __name__ == "__main__":
    # 2. Define your text samples
    text_to_check = [
        "This is a great post, I totally agree with everything you've said.", # Example 1
        "As a large language model, I am unable to provide personal opinions, but I can offer factual summaries.", # Example 2
        "Just popping in to say this is an incredibly nuanced take. Well done.", # Example 3
        "Yes, I have many Android apps I prefer. However, Google has to force all developers, especially the giants, to adopt adaptive layout. Simply stretching an app to landscape is horrible.",
        "Would have to run all games Windows does",
        "Only if it's a decent Linux distribution as base and not just Android ported to PC.",
        "If even apps in beta worked perfectly and I mean per scale on Android as PC then I would switch. But if you told me my system performance would hold as well and as long as hardware does on PC, and I could upgrade as needed ( Desktop of course ) then yes."
    ]

    # 3. Run detection on the text samples
    print("-" * 30)
    for text in text_to_check:
        result = detect(text)
        
        # 4. Print the result
        print(f"Text: \"{result['text']}\"")
        print(f"Prediction: {result['prediction_label'].upper()}")
        print(f"AI-Generated Confidence: {result['ai_prob']:.2%}")
        print("-" * 30)