import re
from difflib import get_close_matches, SequenceMatcher
import pandas as pd

# Load data
df_tweets = pd.read_pickle('../data/all_tweets.pkl')
df_comments = pd.read_csv('../data-ai-slop-detector/processed_comments.csv')

REMOVE_TOKENS = ('description', 'question', 'prompt', 'text', 'analysis', 'facts', 'fact', 'issue', 'holding', 'conclusion', 'rule', 'citation')

def normalize_text(text: str) -> str:
	if pd.isna(text):
		return ''
	text = str(text).lower()
	for token in REMOVE_TOKENS:
		text = text.replace(f'{token}:', ' ')
		text = text.replace(token, ' ')
	text = re.sub(r'\d+', ' ', text)
	return re.sub(r'[^a-z0-9]', '', text)


# Prepare normalized maps
df_comments['normalized'] = df_comments['comment_content'].map(normalize_text)
normalized_map = df_comments.drop_duplicates('normalized').set_index('normalized')[['comment_content']]
if '' in normalized_map.index:
	normalized_map = normalized_map.drop(index='')
if not normalized_map.index.is_unique:
	raise ValueError('Duplicate normalized keys found in comments dataset.')

# Use the correct column name from all_tweets.pkl
tweet_texts = df_tweets['content']
norm_keys = [key for key in normalized_map.index if key]

matches = {}
match_records = []
unmatched = {}

for idx, text in enumerate(tweet_texts):
	norm = normalize_text(text)
	if norm and norm in normalized_map.index:
		row = normalized_map.loc[norm]
		matches[idx] = {
			'comment_content': row['comment_content']
		}
		match_records.append({
			'tweet_index': idx,
			'original_text': text,
			'matched_comment': row['comment_content'],
			'method': 'exact_alphanum',
			'score': 1.0
		})
	else:
		unmatched[idx] = {'norm': norm, 'original_text': text}

if unmatched:
	for cutoff in [0.98, 0.95, 0.9, 0.85, 0.8, 0.75]:
		if not unmatched:
			break
		for idx, info in list(unmatched.items()):
			candidate_key = info['norm'] or normalize_text(info['original_text'])
			if not candidate_key:
				continue
			candidates = get_close_matches(candidate_key, norm_keys, n=3, cutoff=cutoff)
			if candidates:
				cand = max(candidates, key=lambda key: SequenceMatcher(None, candidate_key, key).ratio())
				score = SequenceMatcher(None, candidate_key, cand).ratio()
				row = normalized_map.loc[cand]
				matches[idx] = {
					'comment_content': row['comment_content']
				}
				match_records.append({
					'tweet_index': idx,
					'original_text': info['original_text'],
					'matched_comment': row['comment_content'],
					'method': f'fuzzy_{cutoff}',
					'score': score
				})
				unmatched.pop(idx)

if unmatched:
	for idx, info in list(unmatched.items()):
		candidate_key = info['norm'] or normalize_text(info['original_text'])
		if not candidate_key:
			continue
		best_key = None
		best_score = 0
		for key in norm_keys:
			score = SequenceMatcher(None, candidate_key, key).ratio()
			if score > best_score:
				best_score = score
				best_key = key
		if best_key and best_score >= 0.65:
			row = normalized_map.loc[best_key]
			matches[idx] = {
				'comment_content': row['comment_content']
			}
			match_records.append({
				'tweet_index': idx,
				'original_text': info['original_text'],
				'matched_comment': row['comment_content'],
				'method': f'fuzzy_best_{best_score:.2f}',
				'score': best_score
			})
			unmatched.pop(idx)

if unmatched:
	missing_count = len(unmatched)
	sample = list(unmatched.values())[:5]
	raise ValueError(f'Unmatched tweets remaining ({missing_count}). Sample entries: {sample}')


# Save results
matched_comments = [matches[idx]['comment_content'] for idx in range(len(tweet_texts))]
result_df = pd.DataFrame({
	'content': tweet_texts,
	'matched_comment_content': matched_comments
})
result_df.to_pickle('./tweet_comment_matched.pkl')

match_summary = pd.DataFrame(match_records)
method_counts = match_summary['method'].value_counts()

print(f'Matched tweets: {len(matches)} / {len(tweet_texts)}')
print('Match methods used:')
print(method_counts)
