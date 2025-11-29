# Social Media Data Processing Pipeline

This repository contains Python scripts for processing social media posts and comments from JSON files, cleaning the text, and extracting metadata into structured CSV files.

## 📁 File Structure

```
data_preparation/
├── process_posts.py          # Process posts and shared utility functions
├── process_comments.py        # Process comments with metadata extraction
├── README.md                  # This file
├── influencer_data/           # Input directory (JSON files)
│   ├── AdamParkhomenko.json
│   ├── AnotherUser.json
│   └── ...
└── outputs/                   # Output directory (CSV files)
    ├── links
    │   └──links.csv
    ├── processed_posts.csv
    └── processed_comments.csv
```

You should get the output folders from huggingface repo: 
https://huggingface.co/datasets/ronanhansel/data-ai-slop-detector

## 🚀 Quick Start

### Installation

Install required dependencies:

For twscrape, refer to its repository for installation instructions: https://github.com/vladkens/twscrape

Then install Python packages:

```bash
pip install emoji unidecode
```

### Processing Posts

```python
from process_posts import process_all_posts

# Process all JSON files in a directory
process_all_posts(
    input_dir="influencer_data",
    output_csv_path="outputs/processed_posts.csv"
)
```

### Processing Comments

```python
from process_comments import process_all_comments

# Process all comments from JSON files
process_all_comments(
    input_dir="influencer_data",
    output_csv_path="outputs/processed_comments.csv"
)
```

## 📊 Output CSV Files

### 1. **processed_posts.csv**

Contains cleaned and processed posts with the following columns:

| Column | Description |
|--------|-------------|
| `author_id` | Username/ID of the post author (from filename) |
| `post_id` | Sequential post number (starts at 1 per author) |
| `post_original_content` | Original unprocessed post text |
| `processed_content` | Cleaned and normalized post text |

**Example:**
```csv
author_id,post_id,post_original_content,processed_content
AdamParkhomenko,1,"RT @someone: Check this out! 🔥 #Breaking","check breaking"
AdamParkhomenko,2,"I can't believe this happened in CA today","not believe happened california today"
```

### 2. **processed_comments.csv**

Contains comments with extensive metadata extraction:

| Column | Description |
|--------|-------------|
| `commenter_id` | Username of the comment author |
| `comment_id` | Unique global comment ID (increments across all files) |
| `parent_id` | ID of parent comment (empty for root comments) |
| `post_id` | ID of the post this comment belongs to |
| `comment_content` | Original unprocessed comment text |
| `cleaned_content` | Cleaned and normalized comment text |
| `num_emojis` | Count of Unicode emojis (e.g., 😂, 🔥) |
| `num_text_emojis` | Count of text emoticons (e.g., :), XD, ಠ_ಠ) |
| `num_caps_words` | Count of ALL CAPS words (min 2 characters) |
| `num_unicode_chars` | Count of non-ASCII Unicode characters |
| `contains_media` | Boolean: contains photo/video links |
| `contains_link` | Boolean: contains any HTTP link |
| `num_tagged_people` | Count of @mentions (excludes post/parent authors) |
| `tagged_grok` | Boolean: mentioned @grok |
| `used_slang` | Boolean: used slang, state abbreviations, or country codes |

**Example:**
```csv
commenter_id,comment_id,parent_id,post_id,comment_content,cleaned_content,num_emojis,num_text_emojis,num_caps_words,num_unicode_chars,contains_media,contains_link,num_tagged_people,tagged_grok,used_slang
JohnDoe,1,,1,"@AdamParkhomenko OMG this is crazy!! 😂😂",oh my god crazy,2,0,1,0,False,False,0,False,True
JaneSmith,2,1,1,"@JohnDoe IKR! Check this link https://t.co/abc",i know right check link,0,0,1,0,False,True,0,False,True
```

## 🔧 Text Cleaning Pipeline

The cleaning process includes:

### 1. **Unicode Normalization**
- Converts accented characters to ASCII (e.g., `política` → `politica`)
- Removes special characters: `( ) [ ] { } ! ? | * ~ ` ^ < > ¡ •`

### 2. **Mention Handling**
- Removes RT (retweet) prefixes
- Removes author's own @mention
- Removes parent author's @mention
- Replaces remaining @mentions with `tag`

### 3. **URL Processing**
- Replaces photo/video links with `media`
- Replaces other links with `link`

### 4. **Hashtag Processing**
- Removes `#` symbol
- Splits camelCase: `#BreakingNews` → `breaking news`
- Splits alphanumeric: `#COVID19` → `covid number`

### 5. **Emoji Handling**
- Removes all Unicode emojis
- Normalizes punctuation (quotes, dashes, ellipsis)

### 6. **Contraction Expansion**
- Negative: `don't` → `do not`, `can't` → `can not`
- Positive: `I'm` → `I`, `you're` → `you` (removes auxiliary verbs)

### 7. **Number Normalization**
- Large numbers: `7k`, `213m` → `number`
- Comma-separated: `1,000` → `number`
- Ordinals: `1st`, `2nd` → `number`
- Years: `2024` → `year`

### 8. **Abbreviation Expansion**
- US States: `CA` → `california`, `NY` → `new york`
- Countries: `UK` → `united kingdom`, `US` → `united states`

### 9. **Slang Translation**
- `lol` → `laugh out loud`
- `omg` → `oh my god`
- `btw` → `by the way`
- 100+ slang terms supported

### 10. **Final Cleanup**
- Removes stop words (`the`, `a`, `is`, `was`, etc.)
- Normalizes repeated characters: `soooooo` → `soo`
- Replaces negative terms with `not`
- Converts to lowercase
- Trims whitespace

## 📝 Usage Examples

### Single File Processing

```python
from process_posts import process_posts

# Process a single post file
process_posts(
    json_file_path="influencer_data/AdamParkhomenko.json",
    output_csv_path="outputs/adam_posts.csv"
)
```

### Batch Processing

```python
from process_posts import process_all_posts
from process_comments import process_all_comments

# Process all posts
process_all_posts("influencer_data", "outputs/all_posts.csv")

# Process all comments
process_all_comments("influencer_data", "outputs/all_comments.csv")
```

### Custom Analysis

```python
from process_posts import count_emojis, count_caps_locked_words

text = "OMG this is AMAZING!! 😂🔥"
print(f"Emojis: {count_emojis(text)}")  # Output: 2
print(f"CAPS words: {count_caps_locked_words(text)}")  # Output: 2
```

## 🔍 Metadata Detection Features

### Emoji Detection
- **Unicode emojis**: 😂, 🔥, ❤️
- **Text emoticons**: :), XD, ಠ_ಠ, ¯\_(ツ)_/¯

### Slang Detection
Detects 100+ internet slang terms including:
- Common: `lol`, `lmao`, `omg`, `wtf`
- Modern: `sus`, `goat`, `bussin`, `cap`
- Expressions: `ngl`, `fr`, `tbh`, `imo`

### Link Classification
- **Media links**: Twitter photos/videos, twimg.com
- **Regular links**: Any other HTTP/HTTPS URL

### Tagging Analysis
- Counts @mentions
- Excludes default tags (post author, parent commenter)
- Detects @grok mentions

## 📌 Important Notes

1. **Comment IDs**: Globally unique across all files (starts at 1, increments sequentially)
2. **Post IDs**: Resets to 1 for each author
3. **Parent IDs**: Empty string for root-level comments, integer for replies
4. **Boolean Fields**: Stored as `True`/`False` in CSV
5. **Empty Fields**: Empty string (`""`) used for null/missing values

## 🛠️ Customization

### Adding New Slang Terms

Edit `slang_map` in `process_posts.py`:

```python
slang_map = {
    "lol": "laugh out loud",
    "your_term": "expanded form",
    # ... more terms
}
```

### Adding New State/Country Codes

Edit `us_states` or `countries` dictionaries in `process_posts.py`.

### Modifying Cleaning Rules

Edit the `clean_text_for_posts()` function in `process_posts.py` to adjust the cleaning pipeline.

## 📚 Dependencies

- `emoji`: Unicode emoji detection and removal
- `unidecode`: Unicode to ASCII transliteration
- `re`: Regular expressions (built-in)
- `json`: JSON parsing (built-in)
- `csv`: CSV writing (built-in)
- `pathlib`: File path handling (built-in)

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'emoji'`
```bash
pip install emoji
```

**Issue**: `ModuleNotFoundError: No module named 'unidecode'`
```bash
pip install unidecode
```

**Issue**: CSV shows garbled characters
- Ensure files are opened with `encoding="utf-8"`

**Issue**: Comment IDs not unique
- Make sure to run `process_all_comments()` in a single execution
- Don't process files separately then combine

## 📄 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Feel free to submit issues or pull requests for improvements!

---

**Last Updated**: November 2025