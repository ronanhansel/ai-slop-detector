import tweetnlp

def sentiment_analysis(text):
    """
    Analyzes the sentiment of the given text using TweetNLP.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    model = tweetnlp.load_model('sentiment')
    result = model.sentiment(text, return_probability=True)
    return result

def irony_detection(text):
    """
    Detects irony in the given text using TweetNLP.

    Args:
        text (str): The text to analyze.
    Returns:
        dict: A dictionary containing the irony label and score.
    """
    model = tweetnlp.load_model('irony')
    result = model.irony(text, return_probability=True)
    return result

def hatespeech_detection(text):
    """
    Detects hate speech in the given text using TweetNLP.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the hate speech label and score.
    """
    model = tweetnlp.load_model('hate')
    result = model.hate(text, return_probability=True)
    return result

def offensive_language_detection(text):
    """
    Detects offensive language in the given text using TweetNLP.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the offensive language label and score.
    """
    model = tweetnlp.load_model('offensive')
    result = model.offensive(text, return_probability=True)
    return result

if __name__ == "__main__":
    sample_text = "FAFO. If you threaten or lay hands on our law enforcement officers we will hunt you down and you will find out, really quick. We’ll see you cowards soon."
    sentiment_result = sentiment_analysis(sample_text)
    hatespeech_result = hatespeech_detection(sample_text)
    offensive_result = offensive_language_detection(sample_text)

    print(f"Sentiment Analysis: {sentiment_result}")
    print(f"Irony Detection: {irony_detection(sample_text)}")
    print(f"Hate Speech Detection: {hatespeech_result}")
    print(f"Offensive Language Detection: {offensive_result}")