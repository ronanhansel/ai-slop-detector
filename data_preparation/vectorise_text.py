import pandas as pd
from empath import Empath
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
import re

tqdm.pandas()

# ============================
# 1. Load CSV
# ============================
print("🔹 Loading CSV...")
df = pd.read_csv("/content/processed_all_comments.csv")
print(f"   → Loaded {len(df)} rows")

# ============================
# 2. Load Empath
# ============================
print("🔹 Initializing Empath...")
lexicon = Empath()

def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())

def empath_vectorize(text):
    if pd.isna(text) or text.strip() == "":
        return {cat: 0 for cat in lexicon.cats}
    return lexicon.analyze(text, normalize=True)

# ============================
# 3. Apply Empath with progress bar
# ============================
print("🔹 Vectorizing text with Empath...")
empath_vectors = df["cleaned_content_LIWC"].fillna("").progress_apply(empath_vectorize)
empath_df = pd.DataFrame(list(empath_vectors))
print("   → Empath vectorization done.")

# ============================
# 4. LSA (TF-IDF → SVD)
# ============================
print("🔹 Running TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df["cleaned_content_LSA"].fillna(""))
print("   → TF-IDF done.")

empath_dim = empath_df.shape[1]
lsa_dim = max(50, 256 - empath_dim)

print(f"🔹 Running SVD (components = {lsa_dim})...")
svd = TruncatedSVD(n_components=lsa_dim)
X_lsa = svd.fit_transform(X_tfidf)
lsa_df = pd.DataFrame(X_lsa, columns=[f"lsa_dim_{i}" for i in range(lsa_dim)])
print("   → SVD done.")

# ============================
# 5. Concat Empath + LSA
# ============================
print("🔹 Concatenating vectors...")
final_vector = pd.concat([empath_df, lsa_df], axis=1)

print("🔹 Converting each row to list vector...")
df["vectorise_text"] = final_vector.progress_apply(lambda row: row.values.tolist(), axis=1)

# ============================
# 6. Export CSV
# ============================
print("🔹 Exporting CSV files...")
df_drop = df.drop(columns=["cleaned_content_LIWC", "cleaned_content_LSA"])
df_drop.to_csv("final_vectorized.csv", index=False)
empath_df.to_csv("empath_features.csv", index=False)
lsa_df.to_csv("lsa_features.csv", index=False)

print("✅ All tasks completed!")
print("- final_vectorized.csv")
print("- empath_features.csv")
print("- lsa_features.csv")
