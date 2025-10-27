from transformers import pipeline

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using a pre-trained sentiment analysis model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The sentiment analysis result containing the label (negative, neutral, positive) and score.
    """
    sentiment_task = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return sentiment_task(text)

if __name__ == "__main__":
    sample_text = "FAFO. If you threaten or lay hands on our law enforcement officers we will hunt you down and you will find out, really quick. We’ll see you cowards soon. https://t.co/KmxCp2IEKc"
    result = analyze_sentiment(sample_text)
    print(f"Sentiment analysis result for the text: \"{sample_text}\"")
    print(result)