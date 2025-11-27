import json
import re
from pathlib import Path
import csv
import emoji
from unidecode import unidecode
from typing import Dict, List, Tuple

def count_emojis(text: str) -> int:
    """Count Unicode emojis in text."""
    return emoji.emoji_count(text)

def count_text_emojis(text: str) -> int:
    """Count text-based emoticons and kaomoji."""
    text_emoji_patterns = [
        r"¯\\_\(ツ\)_/¯", r"\(ツ\)", r"ಠ_ಠ", r"ಠ益ಠ", r"ʘ‿ʘ",
        r"\(╯°□°\)╯︵ ┻━┻", r"┬─┬ノ\( º _ ºノ\)", r"ᕕ\( ᐛ \)ᕗ",
        r"\(^_^\)", r"\^_\^", r"＼\(^o^\)／", r"٩\(◕‿◕｡\)۶",
        r"\(づ｡◕‿‿◕｡\)づ", r"ლ\(ಠ益ಠლ\)", r"¯\\_\(シ\)_/¯",
        r"\(⌐■_■\)", r"\(╥_╥\)", r"\(T_T\)", r"\._.+", r";_;+",
        r"T_T", r"o_o", r"O_O", r">_<", r"\^_\^", r"XD", r"xD",
        r":D", r"=D", r":\]", r":\[", r":\(", r":\)", r";\)", r";D"
    ]
    count = 0
    for pattern in text_emoji_patterns:
        count += len(re.findall(pattern, text))
    return count

def count_caps_locked_words(text: str) -> int:
    """Count words that are all uppercase (min 2 chars)."""
    words = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(words)

def count_unicode_chars(text: str) -> int:
    """Count non-ASCII Unicode characters."""
    return sum(1 for c in text if ord(c) > 127)

def contains_media_link(text: str) -> bool:
    """Check if text contains media links."""
    media_patterns = [r'/photo/', r'/video/', r'twimg\.com', r'pbs\.twimg\.com']
    for pattern in media_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def contains_link(text: str) -> bool:
    """Check if text contains any HTTP links."""
    return bool(re.search(r'https?://', text))

def count_tagged_people(text: str, exclude_authors: List[str] = None) -> int:
    """Count @mentions excluding specific authors."""
    mentions = re.findall(r'@(\w+)', text)
    if exclude_authors:
        mentions = [m for m in mentions if m.lower() not in [a.lower() for a in exclude_authors]]
    return len(mentions)

def tagged_grok(text: str) -> bool:
    """Check if @grok is mentioned."""
    return bool(re.search(r'@grok\b', text, re.IGNORECASE))

def used_slang_or_abbreviation(original: str, cleaned: str) -> bool:
    """Check if slang, state abbreviations, or country codes were used."""
    # Define all slang, state abbreviations, and country codes
    slang = {
        "lol", "lmao", "lmfao", "lmk", "rofl", "omg", "wtf", "btw", "imo", "imho",
        "tbh", "smh", "fyi", "idk", "irl", "brb", "af", "rn", "dm", "rt", "ngl",
        "fr", "bc", "tho", "thru", "ppl", "plz", "thx", "u", "ur", "y", "r", "b4",
        "gr8", "m8", "2day", "2nite", "tmr", "tmrw", "asap", "aka", "fomo", "goat",
        "sus", "lowkey", "highkey", "gtfo", "stfu", "jk", "ikr", "tfw", "mfw",
        "yolo", "fam", "bro", "sis", "gonna", "wanna", "gotta", "kinda", "sorta",
        "dunno", "cuz", "w/", "w/o", "pov", "rip", "ftw", "gg", "wp", "ez", "op",
        "lit", "salty", "savage", "flex", "stan", "simp", "based", "ratio", "copium",
        "hopium", "oof", "yeet", "bet", "cap", "bussin", "periodt", "tea", "shade",
        "slay", "queen", "king", "w", "l"
    }
    
    us_states = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
        "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
        "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
        "WI", "WY", "DC"
    }
    
    countries = {
        "US", "USA", "UK", "GB", "GBR", "CA", "CAN", "CN", "CHN", "JP", "JPN",
        "DE", "DEU", "FR", "FRA", "IT", "ITA", "ES", "ESP", "AU", "AUS", "BR",
        "BRA", "IN", "IND", "RU", "RUS", "MX", "MEX", "KR", "KOR", "AR", "ARG",
        "ZA", "ZAF", "SA", "SAU", "TR", "TUR", "PL", "POL", "NL", "NLD", "BE",
        "BEL", "SE", "SWE", "NO", "NOR", "DK", "DNK", "FI", "FIN", "IE", "IRL",
        "PT", "PRT", "GR", "GRC", "CH", "CHE", "AT", "AUT", "NZ", "NZL", "SG",
        "SGP", "HK", "HKG", "IL", "ISR", "EG", "EGY", "NG", "NGA", "KE", "KEN",
        "PH", "PHL", "TH", "THA", "VN", "VNM", "MY", "MYS", "ID", "IDN", "PK",
        "PAK", "BD", "BGD", "UA", "UKR"
    }
    
    # Check if any slang/abbreviation is in the original text
    original_words = set(re.findall(r'\b\w+\b', original.lower()))
    return bool(
        original_words & slang or
        original_words & {s.lower() for s in us_states} or
        original_words & {c.lower() for c in countries}
    )

def clean_text_for_posts(content: str, author: str = None) -> str:
    """Clean text content for posts (simplified version)."""
    if not content:
        return ""
    
    # Load dictionaries
    us_states = {
        "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
        "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
        "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
        "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
        "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
        "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
        "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
        "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
        "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
        "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
        "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
        "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
        "WI": "wisconsin", "WY": "wyoming", "DC": "washington dc"
    }
    
    countries = {
        "US": "united states", "USA": "united states",
        "UK": "united kingdom", "GB": "united kingdom", "GBR": "united kingdom",
        "CA": "canada", "CAN": "canada",
        "CN": "china", "CHN": "china",
        "JP": "japan", "JPN": "japan",
        "DE": "germany", "DEU": "germany",
        "FR": "france", "FRA": "france",
        "IT": "italy", "ITA": "italy",
        "ES": "spain", "ESP": "spain",
        "AU": "australia", "AUS": "australia",
        "BR": "brazil", "BRA": "brazil",
        "IN": "india", "IND": "india",
        "RU": "russia", "RUS": "russia",
        "MX": "mexico", "MEX": "mexico",
        "KR": "south korea", "KOR": "south korea",
    }
    
    slang_map = {
        "lol": "laugh out loud", "lmao": "laughing", "lmfao": "laughing",
        "haha": "laughing", "lmk": "let me know", "rofl": "laughing",
        "omg": "oh my god", "wtf": "what the fuck", "btw": "by the way",
        "imo": "in my opinion", "imho": "in my humble opinion",
        "tbh": "to be honest", "smh": "shaking my head",
        "fyi": "for your information", "idk": "i do not know",
        "irl": "in real life", "brb": "be right back", "af": "as fuck",
        "rn": "right now", "dm": "direct message", "rt": "retweet",
        "ngl": "not gonna lie", "fr": "for real", "bc": "because",
        "tho": "though", "thru": "through", "ppl": "people",
        "plz": "please", "thx": "thanks", "u": "you", "ur": "your",
        "y": "why", "r": "are", "b4": "before", "gr8": "great",
        "gonna": "going to", "wanna": "want to", "gotta": "got to",
    }
    
    contractions_negative = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
        "couldn't": "could not", "can't": "can not", "cannot": "can not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not", "haven't": "have not",
        "hadn't": "had not",
    }
    
    contractions_positive = {
        "i'm": "i", "you're": "you", "he's": "he", "she's": "she",
        "it's": "it", "we're": "we", "they're": "they",
        "i'll": "i", "you'll": "you", "he'll": "he", "she'll": "she",
        "it'll": "it", "we'll": "we", "they'll": "they",
    }
    
    remove_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "could", "may", "might", "must",
        "can", "shall", "this", "that", "these", "those"
    }
    
    negative_terms = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere"
    }
    
    # Unicode conversion
    content = unidecode(content)
    
    # Remove specific characters
    chars_to_strip = ['(', ')', '[', ']', '{', '}', '!', '?', '|', '*', '~', '`', '^', '<', '>', '¡', '•']
    for char in chars_to_strip:
        content = content.replace(char, '')
    
    # Remove leading/trailing special chars
    content = re.sub(r'\b[-+=_]+', '', content)
    content = re.sub(r'[-+=_]+\b', '', content)
    
    # Remove RT
    content = re.sub(r'^RT\s+@\w+:\s*', '', content, flags=re.IGNORECASE)
    
    # Remove author's own mention
    if author:
        content = re.sub(rf'@{re.escape(author)}\s*', '', content, flags=re.IGNORECASE)
    
    # Remove leading @mentions
    content = re.sub(r'^(@\w+\s*)+', '', content)
    
    # Replace @mentions with "tag"
    content = re.sub(r'@\w+', 'tag', content)
    
    # Replace links
    content = re.sub(r'https?://[^\s]+', lambda m: "media" if any(x in m.group(0) for x in ['/photo/', '/video/', 'twimg.com']) else "link", content)
    
    # Process hashtags
    def process_hashtag(match):
        tag = match.group(1)
        spaced = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', tag)
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', spaced)
        return spaced.lower()
    
    content = re.sub(r'#(\w+)', process_hashtag, content)
    
    # Remove emojis
    content = emoji.replace_emoji(content, replace='')
    
    # Normalize quotes and dashes
    content = re.sub(r"[\u2019\u0060\u02BC\u00B4\u2018]", "'", content)
    content = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', content)
    content = re.sub(r"[\u2014\u2013\u2010]", "-", content)
    content = re.sub(r"\u2026|\.{2,}", " ", content)
    content = content.replace('"', '').replace("'", '')
    
    # Handle contractions
    words = content.split()
    expanded_words = []
    
    for word in words:
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w\s\'-]', '', word_lower)
        
        if word_clean in contractions_negative:
            expanded_words.append(contractions_negative[word_clean])
        elif word_clean in contractions_positive:
            expanded_words.append(contractions_positive[word_clean])
        else:
            expanded_words.append(word)
    
    content = ' '.join(expanded_words)
    
    # Replace numbers
    content = re.sub(r'\b\d+\.?\d*[kmbtKMBT]\b', 'number', content)
    content = re.sub(r'\b\d{1,3}(,\d{3})+(\.\d+)?\b', 'number', content)
    content = re.sub(r'\b\d+(st|nd|rd|th)\b', 'number', content)
    
    # Expand abbreviations
    for abbr, full_name in us_states.items():
        content = re.sub(rf'\b{abbr}\b', full_name, content, flags=re.IGNORECASE)
    
    for code, full_name in sorted(countries.items(), key=lambda x: len(x[0]), reverse=True):
        content = re.sub(rf'\b{code}\b', full_name, content, flags=re.IGNORECASE)
    
    content = re.sub(r'\b\d+\b', 'number', content)
    
    # Normalize repeated chars
    content = re.sub(r'(.)\1{2,}', r'\1\1', content)
    
    # Translate slang
    words = content.split()
    translated_words = []
    
    for word in words:
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w\s-]', '', word_lower)
        
        if word_clean in slang_map:
            translated_words.append(slang_map[word_clean])
        else:
            translated_words.append(word)
    
    content = ' '.join(translated_words)
    
    # Final cleanup
    words = content.split()
    cleaned_words = []
    
    for word in words:
        word_lower = word.lower()
        
        if not all(ord(c) < 128 for c in word):
            continue
        
        if word_lower in negative_terms:
            cleaned_words.append("not")
        elif word_lower not in remove_words:
            cleaned_word = re.sub(r'^[^\w]+|[^\w]+$', '', word_lower)
            cleaned_word = re.sub(r'[^\w\s-]', '', cleaned_word)
            
            if cleaned_word and cleaned_word not in remove_words:
                cleaned_words.append(cleaned_word)
    
    content = ' '.join(cleaned_words)
    content = re.sub(r'\s+', ' ', content).strip()
    content = content.lower()
    
    return content

def process_posts(json_file_path: str, output_csv_path: str) -> None:
    """
    Process posts from a JSON file and save to CSV.
    
    Args:
        json_file_path: Path to input JSON file containing posts
        output_csv_path: Path to output CSV file
    """
    json_file = Path(json_file_path)
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get author from filename
    author_id = json_file.stem
    
    # Process posts
    rows = []
    for post_id, post in enumerate(data, start=1):
        original_content = post.get("content", "")
        username = post.get("username", "")
        
        # Clean content
        processed_content = clean_text_for_posts(original_content, author=username)
        
        rows.append({
            "author_id": author_id,
            "post_id": post_id,
            "post_original_content": original_content,
            "processed_content": processed_content
        })
    
    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["author_id", "post_id", "post_original_content", "processed_content"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Processed {len(rows)} posts from {json_file.name}")
    print(f"   Saved to: {output_file}")

def process_all_posts(input_dir: str, output_csv_path: str) -> None:
    """
    Process all post JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_csv_path: Path to output CSV file
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    all_rows = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        author_id = json_file.stem
        
        for post_id, post in enumerate(data, start=1):
            original_content = post.get("content", "")
            username = post.get("username", "")
            processed_content = clean_text_for_posts(original_content, author=username)
            
            all_rows.append({
                "author_id": author_id,
                "post_id": post_id,
                "post_original_content": original_content,
                "processed_content": processed_content
            })
    
    # Write all to CSV
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["author_id", "post_id", "post_original_content", "processed_content"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✅ Total posts processed: {len(all_rows)}")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_directory = "influencer_data"
    output_csv = "outputs/processed_posts.csv"
    
    process_all_posts(input_directory, output_csv)