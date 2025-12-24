"""
Hypothesis Testing for AI Slop Detection
Dual-Stage Inference Engine Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data(path='data-ai-slop-detector/final_detection_processed.pkl'):
    """Load data and compute RBI metric."""
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} comments")
    
    # Convert label to numeric (LABEL_0 -> 0, LABEL_1 -> 1)
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'LABEL_0': 0, 'LABEL_1': 1})
        # If mapping fails, try extracting digit
        if df['label'].isna().any():
            df['label'] = df['label'].astype(str).str.extract('(\d+)').astype(int)
    
    # Convert sentiment_label to numeric for sentiment score
    if 'sentiment_label' in df.columns and df['sentiment_label'].dtype == 'object':
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        df['sentiment_direction'] = df['sentiment_label'].map(sentiment_map).fillna(0)
    else:
        df['sentiment_direction'] = 0
    
    # Extract emotion vector components
    df['s'] = df['sentiment_prob'] * df['sentiment_direction']  # Sentiment with direction
    df['h'] = df['hate_prob']
    df['o'] = df['offensive_prob']
    df['i'] = df['irony_prob']
    
    # Extract rage from empath embedding (assume first element or fallback)
    if 'empath_embedding' in df.columns:
        try:
            df['r'] = df['empath_embedding'].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
        except:
            df['r'] = df['o']  # Fallback
    else:
        df['r'] = df['o']
    
    # Replace inf and NaN
    df['r'] = df['r'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate RBI = (r + o + h) × (1 - |s|)
    df['RBI'] = (df['r'] + df['o'] + df['h']) * (1 - np.abs(df['s']))
    df['RBI'] = df['RBI'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Identify root comments (no parent or parent == post)
    df['is_root'] = df['parent_id'].isna() | (df['parent_id'] == df['post_id'])
    
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"RBI range: [{df['RBI'].min():.4f}, {df['RBI'].max():.4f}]")
    
    return df


def correlation_analysis(df):
    """Analyze correlations between emotion features and AI confidence."""
    results = {'hypothesis': 'CORRELATION', 'title': 'Feature-Confidence Correlations'}
    
    # Select features for correlation
    features = ['s', 'h', 'o', 'i', 'r', 'RBI', 'ai_confidence']
    corr_data = df[features].copy()
    
    # Compute Pearson & Spearman correlations
    pearson_corr = corr_data.corr(method='pearson')
    spearman_corr = corr_data.corr(method='spearman')
    
    # Extract correlations with AI confidence
    ai_pearson = pearson_corr['ai_confidence'].drop('ai_confidence').sort_values(ascending=False)
    ai_spearman = spearman_corr['ai_confidence'].drop('ai_confidence').sort_values(ascending=False)
    
    # P-values for Pearson
    from scipy.stats import pearsonr
    p_values = {}
    for col in ['s', 'h', 'o', 'i', 'r', 'RBI']:
        valid = df[[col, 'ai_confidence']].dropna()
        if len(valid) > 2:
            _, p_val = pearsonr(valid[col], valid['ai_confidence'])
            p_values[col] = p_val
    
    # Feature importance (absolute correlation magnitude)
    abs_corr = ai_pearson.abs().sort_values(ascending=False)
    
    results['pearson'] = ai_pearson
    results['spearman'] = ai_spearman
    results['p_values'] = p_values
    results['abs_magnitude'] = abs_corr
    results['strongest'] = ai_pearson.idxmax()
    results['strongest_r'] = ai_pearson.max()
    
    return results

def plot_correlations(df, corr_results):
    """Visualize correlation matrix and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of all correlations
    features = ['s', 'h', 'o', 'i', 'r', 'RBI', 'ai_confidence']
    corr_matrix = df[features].corr(method='pearson')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Feature Correlation Matrix')
    
    # Bar plot: Feature importance for AI confidence
    importance = corr_results['abs_magnitude'].sort_values(ascending=True)
    colors = ['green' if corr_results['pearson'][feat] > 0 else 'red' for feat in importance.index]
    axes[1].barh(range(len(importance)), importance.values, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(importance)))
    axes[1].set_yticklabels(importance.index)
    axes[1].set_xlabel('|Pearson Correlation| with AI Confidence')
    axes[1].set_title('Feature Importance (Absolute Correlation)')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('data-ai-slop-detector/correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def report_correlations(corr_results):
    """Generate correlation report section."""
    report = f"""
================================================================================
CORRELATION ANALYSIS: Feature-Confidence Relationships
================================================================================
Question: Which emotional features best predict AI confidence?

Pearson Correlations with AI Confidence (ranked by strength):
"""
    for feat, val in corr_results['pearson'].items():
        p_val = corr_results['p_values'].get(feat, np.nan)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        report += f"  {feat:8s}: r = {val:7.4f}  p = {p_val:.2e}  {sig}\n"
    
    report += f"""
Spearman Correlations (rank-based, robust to outliers):
"""
    for feat, val in corr_results['spearman'].items():
        report += f"  {feat:8s}: rho = {val:7.4f}\n"
    
    report += f"""
STRONGEST PREDICTOR: {corr_results['strongest']} (r = {corr_results['strongest_r']:.4f})

Interpretation:
- Positive correlation: Feature increases with AI confidence (bot-like)
- Negative correlation: Feature decreases with AI confidence (human-like)
- *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

================================================================================
"""
    return report

# ============================================================================
# HYPOTHESIS A: AI Slop vs Human Rage-Bait
# ============================================================================

def hypothesis_a(df):
    """Weighted Mann-Whitney U Test for RBI differences."""
    results = {'hypothesis': 'A', 'title': 'AI Slop vs Human Rage-Bait'}
    
    thresholds = [0.5, 0.7, 0.9]
    tests = []
    
    for thresh in thresholds:
        bots = df[(df['label'] == 1) & (df['ai_confidence'] >= thresh)]['RBI']
        humans = df[(df['label'] == 0) & (df['ai_confidence'] >= thresh)]['RBI']
        
        if len(bots) > 10 and len(humans) > 10:
            u_stat, p_val = mannwhitneyu(bots, humans, alternative='greater')
            tests.append({
                'threshold': thresh,
                'n_bots': len(bots), 'n_humans': len(humans),
                'bot_mean': bots.mean(), 'human_mean': humans.mean(),
                'u_stat': u_stat, 'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    results['tests'] = pd.DataFrame(tests)
    results['conclusion'] = 'SUPPORTED' if all(t['significant'] for t in tests) else 'PARTIAL/NOT SUPPORTED'
    return results

# ============================================================================
# HYPOTHESIS B: Trigger Effect (Interaction Analysis)
# ============================================================================

def hypothesis_b(df):
    """Analyze if bot comments trigger higher emotional variance in replies."""
    results = {'hypothesis': 'B', 'title': 'Trigger Effect Analysis'}
    
    # Get root comments
    roots = df[df['is_root']].copy()
    
    # For each root, calculate EV of replies
    root_ev = []
    for _, root in roots.iterrows():
        replies = df[df['parent_id'] == root['comment_id']]
        if len(replies) >= 2:
            ev = replies['sentiment_prob'].var() + replies['offensive_prob'].var()
            root_ev.append({
                'comment_id': root['comment_id'],
                'label': root['label'],
                'ai_confidence': root['ai_confidence'],
                'RBI': root['RBI'],
                'reply_EV': ev,
                'n_replies': len(replies)
            })
    
    if not root_ev:
        results['error'] = 'Insufficient reply data'
        return results
    
    df_ev = pd.DataFrame(root_ev)
    
    # Compare EV for bot vs human roots (high RBI only)
    rbi_thresh = df_ev['RBI'].quantile(0.75)
    high_rbi = df_ev[df_ev['RBI'] >= rbi_thresh]
    
    bot_ev = high_rbi[high_rbi['label'] == 1]['reply_EV']
    human_ev = high_rbi[high_rbi['label'] == 0]['reply_EV']
    
    if len(bot_ev) > 5 and len(human_ev) > 5:
        u_stat, p_val = mannwhitneyu(bot_ev, human_ev, alternative='greater')
        results['test'] = {
            'bot_mean_ev': bot_ev.mean(), 'human_mean_ev': human_ev.mean(),
            'u_stat': u_stat, 'p_value': p_val, 'significant': p_val < 0.05
        }
        results['conclusion'] = 'SUPPORTED' if p_val < 0.05 else 'NOT SUPPORTED'
    else:
        results['error'] = f'Insufficient high-RBI samples (bots={len(bot_ev)}, humans={len(human_ev)})'
    
    results['data'] = df_ev
    return results

# ============================================================================
# HYPOTHESIS C: Side-Specific Polarization (Two-Way ANOVA)
# ============================================================================

def hypothesis_c(df, influencer_leaning=None):
    """Two-Way ANOVA for political lean × bot interaction."""
    results = {'hypothesis': 'C', 'title': 'Side-Specific Polarization'}
    
    # Check for influencer leaning data
    if influencer_leaning is None:
        # Try to load from file
        try:
            influencer_leaning = pd.read_csv('influencer_leaning.csv')
        except FileNotFoundError:
            results['error'] = 'Influencer political leaning data required'
            results['note'] = 'File: influencer_leaning.csv with columns: author_id, leaning (Left/Right/Neutral)'
            return results
    
    # Load post_id mappings from processed posts
    try:
        posts = pd.read_csv('data_preparation/outputs/processed_posts.csv')
    except FileNotFoundError:
        results['error'] = 'Post mapping file not found'
        results['note'] = 'File: data_preparation/outputs/processed_posts.csv with columns: post_id, author_id'
        return results
    
    # Ensure author_id is string type for consistency
    df = df.copy()
    posts['author_id'] = posts['author_id'].astype(str)
    influencer_leaning['author_id'] = influencer_leaning['author_id'].astype(str)
    
    # Merge posts with leaning data by author_id
    posts_leaning = posts.merge(influencer_leaning, on='author_id', how='inner')
    
    if len(posts_leaning) == 0:
        results['error'] = 'No matching authors between posts and leaning data'
        return results
    
    print(f"Matched {len(posts_leaning)} posts to {len(posts_leaning['author_id'].unique())} influencers")
    
    # Merge comments with post leaning data
    df['post_id'] = df['post_id'].astype(str)
    posts_leaning['post_id'] = posts_leaning['post_id'].astype(str)
    
    df_merged = df.merge(posts_leaning[['post_id', 'leaning']], 
                         left_on='post_id', right_on='post_id', how='inner')
    
    # Filter for only Left/Right for a cleaner 2x2 ANOVA, but keep Neutral for group stats
    df_anova = df_merged[df_merged['leaning'].isin(['Left', 'Right'])].copy()
    
    if len(df_anova) < 100:
        results['error'] = f'Insufficient data for Left/Right leaning groups ({len(df_anova)} rows)'
        return results
    
    print(f"Using {len(df_anova)} comments from Left/Right leaning posts for ANOVA")
    
    # Map label to a readable string for the model formula
    df_anova['commenter_type'] = df_anova['label'].map({0: 'Human', 1: 'Bot'})

    # Perform Two-Way ANOVA
    try:
        print("Training Two-Way ANOVA model...")
        model = ols('RBI ~ C(leaning) + C(commenter_type) + C(leaning):C(commenter_type)', data=df_anova).fit()
        print(model.summary())
        anova_results = anova_lm(model, typ=2)
        results['anova_table'] = anova_results
        
        # Check interaction significance
        interaction_p = anova_results.loc['C(leaning):C(commenter_type)', 'PR(>F)']
        interaction_f = anova_results.loc['C(leaning):C(commenter_type)', 'F']
        
        # Store extracted stats for reporting
        results['anova'] = {
            'f_stat': interaction_f,
            'p_value': interaction_p,
            'significant': interaction_p < 0.05
        }
        results['interaction'] = {'p_value': interaction_p, 'significant': interaction_p < 0.05}
        results['conclusion'] = 'SIGNIFICANT INTERACTION FOUND' if interaction_p < 0.05 else 'NO SIGNIFICANT INTERACTION'

    except Exception as e:
        results['error'] = f"ANOVA failed: {e}"
        return results

    # Calculate group statistics for all leanings
    groups_data = {}
    for leaning in ['Left', 'Right', 'Neutral']:
        for label in [0, 1]:
            key = f'{leaning.lower()}_{"bot" if label==1 else "human"}'
            subset = df_merged[(df_merged['leaning']==leaning) & (df_merged['label']==label)]['RBI']
            if len(subset) > 0:
                groups_data[key] = {'mean': subset.mean(), 'n': len(subset), 'std': subset.std()}

    results['groups'] = groups_data
    results['data'] = df_merged
    
    return results

# ============================================================================
# HYPOTHESIS D: Super-Spreaders (Probabilistic Bot Count)
# ============================================================================

def hypothesis_d(df, top_pct=0.01):
    """Identify top rage-baiters and calculate probabilistic bot count."""
    results = {'hypothesis': 'D', 'title': 'Super-Spreaders Analysis'}
    
    # Ensure label is numeric
    if df['label'].dtype == 'object':
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Aggregate by user
    user_stats = df.dropna(subset=['label']).groupby('commenter_id').agg({
        'RBI': 'mean',
        'label': 'mean',
        'ai_confidence': 'mean',
        'comment_id': 'count'
    }).rename(columns={'comment_id': 'n_comments'}).reset_index()
    
    # Get top 1% by RBI
    n_top = max(int(len(user_stats) * top_pct), 10)
    top_users = user_stats.nlargest(n_top, 'RBI')
    
    # Probabilistic bot count
    prob_bot_count = top_users['ai_confidence'].sum()
    pct_synthetic = prob_bot_count / n_top * 100
    
    # Hard threshold comparison
    hard_bot_count = (top_users['label'] >= 0.5).sum()
    
    # Baseline
    baseline_pct = df['label'].mean() * 100
    
    results['stats'] = {
        'n_top_users': n_top,
        'prob_bot_count': prob_bot_count,
        'pct_synthetic': pct_synthetic,
        'hard_bot_count': hard_bot_count,
        'hard_pct': hard_bot_count / n_top * 100,
        'baseline_bot_pct': baseline_pct,
        'enrichment': pct_synthetic / baseline_pct if baseline_pct > 0 else np.nan
    }
    results['top_users'] = top_users
    results['conclusion'] = f'{pct_synthetic:.1f}% of top rage-baiters are likely bots ({results["stats"]["enrichment"]:.2f}x baseline)'
    
    return results

def generate_report(results_a, results_b, results_c, results_d, df):
    """Generate comprehensive report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
================================================================================
                    AI SLOP DETECTION - HYPOTHESIS TESTING REPORT
                              Generated: {timestamp}
================================================================================

DATASET SUMMARY
---------------
Total Comments: {len(df):,}
Bot Comments: {(df['label']==1).sum():,} ({100*(df['label']==1).mean():.1f}%)
Human Comments: {(df['label']==0).sum():,} ({100*(df['label']==0).mean():.1f}%)
Mean AI Confidence: {df['ai_confidence'].mean():.3f}
Mean RBI: {df['RBI'].mean():.4f} (std: {df['RBI'].std():.4f})

================================================================================
HYPOTHESIS A: {results_a['title']}
================================================================================
Question: Do AI comments have higher Rage-Bait Index than humans?

Sensitivity Analysis (across confidence thresholds):
{results_a['tests'].to_string(index=False) if 'tests' in results_a else 'N/A'}

CONCLUSION: {results_a['conclusion']}

================================================================================
HYPOTHESIS B: {results_b['title']}
================================================================================
Question: Do bot comments trigger more emotional variance in replies?

"""
    if 'test' in results_b:
        report += f"""Results:
- Bot root comments (high RBI) → Mean reply EV: {results_b['test']['bot_mean_ev']:.4f}
- Human root comments (high RBI) → Mean reply EV: {results_b['test']['human_mean_ev']:.4f}
- Mann-Whitney U: {results_b['test']['u_stat']:.2f}, p-value: {results_b['test']['p_value']:.6f}

CONCLUSION: {results_b['conclusion']}
"""
    else:
        report += f"ERROR: {results_b.get('error', 'Unknown error')}\n"

    report += f"""
================================================================================
HYPOTHESIS C: {results_c['title']}
================================================================================
Question: Is polarization amplified differently under Left vs Right influencers?

"""
    if results_c and 'anova' in results_c:
        report += f"""Group Statistics (RBI by Political Lean × Bot/Human):
"""
        groups_df = pd.DataFrame([{'group': k, **v} for k, v in results_c['groups'].items()])
        report += groups_df.to_string(index=False)
        report += f"""

ANOVA Results:
- F-statistic: {results_c['anova']['f_stat']:.4f}
- p-value: {results_c['anova']['p_value']:.6f}
- Significant: {'Yes' if results_c['anova']['significant'] else 'No'} (α=0.05)

CONCLUSION: {results_c['conclusion']}
"""
    else:
        note = results_c.get('note', '') if results_c else 'Create influencer_leaning.csv with columns: post_id, leaning (Left/Right/Neutral)'
        report += f"""NOTE: Political leaning data not available.
{note}

CONCLUSION: SKIPPED - Requires external influencer political leaning data
"""

    report += f"""
================================================================================
HYPOTHESIS D: {results_d['title']}
================================================================================
Question: What percentage of top "rage-baiters" are bots?

Results:
- Analyzed top {results_d['stats']['n_top_users']} users by average RBI
- Probabilistic bot count: {results_d['stats']['prob_bot_count']:.1f}
- Estimated % synthetic: {results_d['stats']['pct_synthetic']:.1f}%
- Hard threshold count: {results_d['stats']['hard_bot_count']} ({results_d['stats']['hard_pct']:.1f}%)
- Baseline bot %: {results_d['stats']['baseline_bot_pct']:.1f}%
- Enrichment factor: {results_d['stats']['enrichment']:.2f}x

CONCLUSION: {results_d['conclusion']}

================================================================================
                                 END OF REPORT
================================================================================
"""
    return report

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df, results_a, results_b, results_c, results_d):
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # A: RBI by label
    bots = df[df['label']==1]['RBI']
    humans = df[df['label']==0]['RBI']
    axes[0,0].boxplot([humans, bots], tick_labels=['Human', 'Bot'])
    axes[0,0].set_ylabel('RBI')
    axes[0,0].set_title('Hypothesis A: RBI Distribution')
    axes[0,0].grid(alpha=0.3)
    
    # B: Trigger effect
    if 'data' in results_b:
        ev_df = results_b['data']
        for label, color in [(0, 'blue'), (1, 'orange')]:
            subset = ev_df[ev_df['label']==label]
            axes[0,1].scatter(subset['RBI'], subset['reply_EV'], alpha=0.5, 
                             label='Human' if label==0 else 'Bot', c=color, s=30)
        axes[0,1].set_xlabel('Root RBI')
        axes[0,1].set_ylabel('Reply Emotional Variance')
        axes[0,1].set_title('Hypothesis B: Trigger Effect')
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)
    
    # C: Side-specific polarization
    if results_c and 'groups' in results_c:
        groups = results_c['groups']
        leanings = ['left', 'right', 'neutral']
        bot_means = []
        human_means = []
        
        for lean in leanings:
            bot_key = f'{lean}_bot'
            human_key = f'{lean}_human'
            bot_means.append(groups.get(bot_key, {}).get('mean', 0))
            human_means.append(groups.get(human_key, {}).get('mean', 0))
        
        x = np.arange(len(leanings))
        width = 0.35
        axes[0,2].bar(x - width/2, human_means, width, label='Human', color='steelblue', alpha=0.8)
        axes[0,2].bar(x + width/2, bot_means, width, label='Bot', color='coral', alpha=0.8)
        axes[0,2].set_ylabel('Mean RBI')
        axes[0,2].set_title('Hypothesis C: RBI by Political Lean')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels([l.capitalize() for l in leanings])
        axes[0,2].legend()
        axes[0,2].grid(alpha=0.3, axis='y')
    
    # D: Top rage-baiters scatter
    if 'top_users' in results_d:
        top = results_d['top_users']
        scatter = axes[1,0].scatter(top['RBI'], top['ai_confidence'], c=top['label'], 
                         cmap='coolwarm', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        axes[1,0].set_xlabel('Average RBI')
        axes[1,0].set_ylabel('AI Confidence')
        axes[1,0].set_title('Hypothesis D: Top Rage-Baiters')
        axes[1,0].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,0], label='Label (0=Human, 1=Bot)')
    
    # D: Composition pie
    stats = results_d['stats']
    colors_pie = ['#ff7f0e', '#1f77b4']
    axes[1,1].pie([stats['prob_bot_count'], stats['n_top_users']-stats['prob_bot_count']], 
                  labels=[f'Bot\n({stats["prob_bot_count"]:.0f})', 
                         f'Human\n({stats["n_top_users"]-stats["prob_bot_count"]:.0f})'], 
                  autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[1,1].set_title(f'Top {stats["n_top_users"]} Rage-Baiters\nComposition (Probabilistic)')
    
    # C: ANOVA p-value visualization
    if results_c and 'anova' in results_c:
        p_val = results_c['anova']['p_value']
        significance = 'Significant' if p_val < 0.05 else 'Not Significant'
        color = 'green' if p_val < 0.05 else 'red'
        axes[1,2].text(0.5, 0.6, f"F = {results_c['anova']['f_stat']:.4f}", 
                      ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1,2].text(0.5, 0.4, f"p = {p_val:.6f}", 
                      ha='center', va='center', fontsize=12)
        axes[1,2].text(0.5, 0.2, significance, 
                      ha='center', va='center', fontsize=12, 
                      fontweight='bold', color=color)
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Hypothesis C: ANOVA Results')
    
    plt.tight_layout()
    plt.savefig('data-ai-slop-detector/hypothesis_testing_results.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Testing Hypothesis A...")
    results_a = hypothesis_a(df)
    
    print("Testing Hypothesis B...")
    results_b = hypothesis_b(df)
    
    print("Testing Hypothesis C...")
    results_c = hypothesis_c(df)
    
    print("Testing Hypothesis D...")
    results_d = hypothesis_d(df)
    
    print("Correlation Analysis...")
    corr_results = correlation_analysis(df)
    
    print("Generating report...")
    report = generate_report(results_a, results_b, results_c, results_d, df)
    report += report_correlations(corr_results)
    
    # Save report
    with open('data-ai-slop-detector/hypothesis_testing_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    
    print("\nGenerating visualizations...")
    create_visualizations(df, results_a, results_b, results_c, results_d)
    plot_correlations(df, corr_results)
    
    print("\nSaved:")
    print("  - data-ai-slop-detector/hypothesis_testing_report.txt")
    print("  - data-ai-slop-detector/hypothesis_testing_results.png")
    print("  - data-ai-slop-detector/correlation_analysis.png")
    
    return df, results_a, results_b, results_c, results_d, corr_results

if __name__ == '__main__':
    df, ra, rb, rc, rd, cor = main()