import json
import re
from pathlib import Path
import csv
import emoji
from unidecode import unidecode
from typing import Dict, List, Tuple, Optional
import contractions
import nltk
from nltk.corpus import words as nltk_words

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
    
    
    # Check if any slang/abbreviation is in the original text
    original_words = set(re.findall(r'\b\w+\b', original.lower()))
    return bool(original_words & slang)

def clean_text_for_posts_LSA(content: str, author: str = None) -> str:
    """Clean text content for posts (simplified version)."""
    if not content:
        return ""
    
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
    content = contractions.fix(content)
    
    # Remove specific characters
    chars_to_strip = ['(', ')', '[', ']', '{', '}', '!', '?', '|', '*', '~', '`', '^', '<', '>', '¡', '•','--','-']
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
    
    
    # Replace numbers
    content = re.sub(r'\b\d+\.?\d*[kmbtKMBT]\b', 'number', content)
    content = re.sub(r'\b\d{1,3}(,\d{3})+(\.\d+)?\b', 'number', content)
    content = re.sub(r'\b\d+(st|nd|rd|th)\b', 'number', content)
    
    # Expand abbreviations
    # for abbr, full_name in us_states.items():
    #     content = re.sub(rf'\b{abbr}\b', full_name, content, flags=re.IGNORECASE)
    
    # for code, full_name in sorted(countries.items(), key=lambda x: len(x[0]), reverse=True):
    #     content = re.sub(rf'\b{code}\b', full_name, content, flags=re.IGNORECASE)
    
    content = re.sub(r'\b\d+\b', '', content)
    
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

        # If token contains digits, attempt special digit/leet handling
        if re.search(r"\d", word):
            handled = _handle_digit_word(word, ENGLISH_WORDS)
            if handled is None:
                # Unable to reconstruct into valid words -> drop token
                continue
            # Expand handled tokens (could be ['abc','number','def'] or ['follow'])
            for t in handled:
                if t == 'number':
                    cleaned_words.append('number')
                    continue
                t_low = t.lower()
                if t_low in negative_terms:
                    cleaned_words.append('not')
                    continue
                if t_low not in remove_words:
                    cleaned_t = re.sub(r'^[^\w]+|[^\w]+$', '', t_low)
                    cleaned_t = re.sub(r'[^\w\s-]', '', cleaned_t)
                    if cleaned_t and cleaned_t not in remove_words:
                        cleaned_words.append(cleaned_t)
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


def _load_english_words_set() -> set:
    """Load English words from nltk corpus. Download if missing.

    Returns a set of lowercase English words for fast membership tests.
    """
    try:
        w = set(word.lower() for word in nltk_words.words())
        return w
    except LookupError:
        # Attempt to download the 'words' corpus then load again
        nltk.download('words')
        w = set(word.lower() for word in nltk_words.words())
        return w


def _handle_digit_word(token: str, english_words_set: set) -> Optional[List[str]]:
    """Handle tokens containing digits using the rules described by the user.

    Rules implemented:
    1. If token is purely digits, return ['number'].
    2. Attempt to replace digits with letters according to a mapping:
       0 -> o or u (try both), 9 -> g, 3 -> e, 6 -> g, 1 -> i
       If any candidate after substitution exists in the English words set, return [candidate].
    3. If substitution doesn't produce a valid word, split token into alpha and digit runs.
       If all alpha runs are valid English words, return the parts with digit runs replaced
       by 'number' (e.g. ['abc', 'number', 'def']).
    4. If none of the above produce a valid word, return None to indicate the token should be removed.

    Note: returned list contains one or more tokens (so we can expand a single input token into
    multiple output tokens like ['abc','number','def']).
    """
    import re

    # If token is purely digits, return 'number'
    if token.isdigit():
        return ['number']

    # Step 1: Attempt substitution mapping for leet-like tokens
    # mapping: 9 -> g, 3 -> e, 6 -> g, and special handling for 0 and 1
    subs_map_base = {'9': 'g', '3': 'e', '6': 'g'}

    def substitute(token_str: str, zero_replacement: str, one_replacement: str) -> str:
        """Replace digits with letters according to the mapping.
        
        Args:
            token_str: The token to substitute
            zero_replacement: What to replace '0' with ('o', 'u', or 'no')
            one_replacement: What to replace '1' with ('i' or 'one')
        """
        out_chars = []
        i = 0
        while i < len(token_str):
            ch = token_str[i]
            if ch == '0':
                out_chars.append(zero_replacement)
            elif ch == '1':
                out_chars.append(one_replacement)
            elif ch in subs_map_base:
                out_chars.append(subs_map_base[ch])
            else:
                out_chars.append(ch)
            i += 1
        return ''.join(out_chars).lower()

    # Generate candidates with priority order
    # Priority 1: 0->o with 1->i (most common leet speak)
    # Priority 2: 0->u with 1->i
    # Priority 3: 0->o with 1->one
    # Priority 4: 0->u with 1->one
    # Priority 5: 0->no with 1->i
    # Priority 6: 0->no with 1->one
    candidates_ordered = [
        substitute(token, 'o', 'i'),
        substitute(token, 'u', 'i'),
        substitute(token, 'o', 'one'),
        substitute(token, 'u', 'one'),
        substitute(token, 'no', 'i'),
        substitute(token, 'no', 'one'),
    ]

    for cand in candidates_ordered:
        if cand in english_words_set:
            return [cand]

    # Step 2: If substitution doesn't work, split into alpha and digit runs
    # and check if all alpha parts are dictionary words
    parts = re.findall(r'[A-Za-z]+|\d+', token)

    alpha_parts = [p for p in parts if p.isalpha()]
    if alpha_parts:
        all_alpha_valid = True
        for ap in alpha_parts:
            if ap.lower() not in english_words_set:
                all_alpha_valid = False
                break

        if all_alpha_valid:
            # Replace digit parts with the word 'number'
            out = []
            for p in parts:
                if p.isdigit():
                    out.append('number')
                else:
                    out.append(p.lower())
            return out

    # No valid reconstruction found
    return None


# Initialize English words set at module import (will download if missing)
try:
    ENGLISH_WORDS = _load_english_words_set()
except Exception:
    # If download fails (e.g., offline), fallback to empty set so token checks simply fail-safe
    ENGLISH_WORDS = set()



def clean_text_for_posts_Empath(content: str, author: str = None) -> str:
    """Clean text content for posts (simplified version)."""
    if not content:
        return ""
    
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

    # Unicode conversion
    content = unidecode(content)
    content = contractions.fix(content)
    
    # Remove specific characters
    chars_to_strip = ['(', ')', '[', ']', '{', '}','|', '*', '~', '`', '^', '<', '>', '¡', '•','--','-']
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
        word_clean = re.sub(r'[^\w\s,.!?]', '', word_lower)

        
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
    content = re.sub(r'\b\d+\b', 'number', content)
    
    # Normalize repeated chars
    content = re.sub(r'(.)\1{2,}', r'\1\1', content)
    
    # Translate slang
    words = content.split()
    translated_words = []
    
    for word in words:
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w\s,.!?]', '', word_lower)

        
        if word_clean in slang_map:
            translated_words.append(slang_map[word_clean])
            continue
        
        else:
            translated_words.append(word_clean)
    
    content = ' '.join(translated_words)
    
    # Final cleanup
    words = content.split()
    cleaned_words = []
    
    for word in words:
        word_lower = word.lower()
        
        if not all(ord(c) < 128 for c in word):
            continue
        # If token contains digits, attempt special digit/leet handling
        if re.search(r"\d", word):
            handled = _handle_digit_word(word, ENGLISH_WORDS)
            if handled is None:
                # Unable to reconstruct -> drop token
                continue
            for t in handled:
                if t == 'number':
                    cleaned_words.append('number')
                    continue
                cleaned_word = re.sub(r'^[^\w.,!?]+|[^\w.,!?]+$', '', t)
                cleaned_word = re.sub(r'[^\w\s\-,.!?]', '', cleaned_word)
                if cleaned_word:
                    cleaned_words.append(cleaned_word)
            continue

        cleaned_word = re.sub(r'^[^\w.,!?]+|[^\w.,!?]+$', '', word_lower)
        cleaned_word = re.sub(r'[^\w\s\-,.!?]', '', cleaned_word)   # thêm lại

        if cleaned_word:
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
        processed_content_Empath = clean_text_for_posts_Empath(original_content, author=username)
        processed_content_LSA = clean_text_for_posts_LSA(original_content, author=username)
        
        rows.append({
            "author_id": author_id,
            "post_id": post_id,
            "post_original_content": original_content,
            "processed_content_Empath": processed_content_Empath,
            "processed_content_LSA": processed_content_LSA
        })
    
    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["author_id", "post_id", "post_original_content", "processed_content_Empath", "processed_content_LSA"]
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
            processed_content_Empath = clean_text_for_posts_Empath(original_content, author=username)
            processed_content_LSA = clean_text_for_posts_LSA(original_content, author=username)
            
            all_rows.append({
                "author_id": author_id,
                "post_id": post_id,
                "post_original_content": original_content,
                "processed_content_Empath": processed_content_Empath,
                "processed_content_LSA": processed_content_LSA
            })
    
    # Write all to CSV
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["author_id", "post_id", "post_original_content", "processed_content_Empath", "processed_content_LSA"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✅ Total posts processed: {len(all_rows)}")
    print(f"   Saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    input_directory = r"E:\Project_DS\ai-slop-detector\data_preparation\influencer_data"
    output_csv = "data_preparation/outputs/processed_posts.csv"
    process_all_posts(input_directory, output_csv)

