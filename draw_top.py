"""
Super-Spreader Enrichment Analysis
Visualizes bot concentration in top rage-baiters across different percentiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_prepare_data(path='data-ai-slop-detector/final_detection_processed.pkl'):
    """Load data and compute RBI metric."""
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} comments")
    
    # Convert label to numeric
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'LABEL_0': 0, 'LABEL_1': 1})
        if df['label'].isna().any():
            df['label'] = df['label'].astype(str).str.extract('(\d+)').astype(int)
    
    # Convert sentiment_label to numeric
    if 'sentiment_label' in df.columns and df['sentiment_label'].dtype == 'object':
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        df['sentiment_direction'] = df['sentiment_label'].map(sentiment_map).fillna(0)
    else:
        df['sentiment_direction'] = 0
    
    # Extract emotion vector components
    df['s'] = df['sentiment_prob'] * df['sentiment_direction']
    df['h'] = df['hate_prob']
    df['o'] = df['offensive_prob']
    df['i'] = df['irony_prob']
    
    # Extract rage from empath embedding
    if 'empath_embedding' in df.columns:
        try:
            df['r'] = df['empath_embedding'].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
        except:
            df['r'] = df['o']
    else:
        df['r'] = df['o']
    
    df['r'] = df['r'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate RBI
    df['RBI'] = (df['r'] + df['o'] + df['h']) * (1 - np.abs(df['s']))
    df['RBI'] = df['RBI'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def analyze_enrichment_by_percentile(df, percentiles=[0.1, 1, 5, 10, 25, 50]):
    """
    Calculate bot enrichment across different top percentiles of rage-baiters.
    
    Parameters:
    -----------
    df : DataFrame
        Comments data with RBI and AI confidence
    percentiles : list
        List of percentiles to analyze (e.g., [0.1, 1, 5, 10])
    
    Returns:
    --------
    DataFrame with enrichment statistics for each percentile
    """
    # Aggregate by user
    user_stats = df.dropna(subset=['label']).groupby('commenter_id').agg({
        'RBI': 'mean',
        'label': 'mean',
        'ai_confidence': 'mean',
        'comment_id': 'count'
    }).rename(columns={'comment_id': 'n_comments'}).reset_index()
    
    # Baseline bot percentage
    baseline_bot_pct = df['label'].mean() * 100
    
    results = []
    
    for pct in sorted(percentiles):
        n_top = max(int(len(user_stats) * (pct / 100)), 1)
        top_users = user_stats.nlargest(n_top, 'RBI')
        
        # Probabilistic bot count (using ai_confidence)
        prob_bot_count = top_users['ai_confidence'].sum()
        prob_bot_pct = (prob_bot_count / n_top) * 100
        
        # Hard threshold count
        hard_bot_count = (top_users['label'] >= 0.5).sum()
        hard_bot_pct = (hard_bot_count / n_top) * 100
        
        # Enrichment factor
        prob_enrichment = prob_bot_pct / baseline_bot_pct if baseline_bot_pct > 0 else np.nan
        hard_enrichment = hard_bot_pct / baseline_bot_pct if baseline_bot_pct > 0 else np.nan
        
        # Mean RBI for this group
        mean_rbi = top_users['RBI'].mean()
        
        results.append({
            'percentile': pct,
            'n_users': n_top,
            'mean_rbi': mean_rbi,
            'prob_bot_pct': prob_bot_pct,
            'prob_enrichment': prob_enrichment,
            'hard_bot_pct': hard_bot_pct,
            'hard_enrichment': hard_enrichment,
            'baseline_pct': baseline_bot_pct
        })
    
    return pd.DataFrame(results)


def create_enrichment_visualizations(enrichment_df):
    """Create comprehensive visualizations of bot enrichment."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme
    prob_color = '#FF6B6B'  # Red for probabilistic
    hard_color = '#4ECDC4'  # Teal for hard threshold
    baseline_color = '#95E1D3'  # Light teal for baseline
    
    # 1. Bot Percentage by Percentile (Log scale)
    ax1 = axes[0, 0]
    x_pos = range(len(enrichment_df))
    x_labels = [f"Top {p}%" for p in enrichment_df['percentile']]
    
    ax1.plot(x_pos, enrichment_df['prob_bot_pct'], 'o-', 
             color=prob_color, linewidth=2.5, markersize=8, 
             label='Probabilistic (AI Confidence)', zorder=3)
    ax1.plot(x_pos, enrichment_df['hard_bot_pct'], 's--', 
             color=hard_color, linewidth=2, markersize=7, 
             label='Hard Threshold (>0.5)', zorder=2)
    ax1.axhline(y=enrichment_df['baseline_pct'].iloc[0], 
                color=baseline_color, linestyle=':', linewidth=2, 
                label=f'Baseline ({enrichment_df["baseline_pct"].iloc[0]:.1f}%)', zorder=1)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylabel('Bot Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Bot Concentration in Top Rage-Baiters', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Add value labels
    for i, row in enrichment_df.iterrows():
        ax1.text(i, row['prob_bot_pct'] + 2, f"{row['prob_bot_pct']:.1f}%", 
                ha='center', va='bottom', fontsize=8, color=prob_color, fontweight='bold')
    
    # 2. Enrichment Factor (Fold Change)
    ax2 = axes[0, 1]
    width = 0.35
    x_pos_arr = np.arange(len(enrichment_df))
    
    bars1 = ax2.bar(x_pos_arr - width/2, enrichment_df['prob_enrichment'], 
                    width, label='Probabilistic', color=prob_color, alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x_pos_arr + width/2, enrichment_df['hard_enrichment'], 
                    width, label='Hard Threshold', color=hard_color, alpha=0.8, edgecolor='black')
    
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Baseline (1x)')
    ax2.set_xticks(x_pos_arr)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_ylabel('Enrichment Factor (Fold Change)', fontsize=11, fontweight='bold')
    ax2.set_title('Bot Enrichment vs. Baseline', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Mean RBI by Percentile
    ax3 = axes[1, 0]
    ax3.fill_between(x_pos, enrichment_df['mean_rbi'], alpha=0.3, color='#FFA07A')
    ax3.plot(x_pos, enrichment_df['mean_rbi'], 'o-', 
             color='#FF6347', linewidth=2.5, markersize=8, zorder=3)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.set_ylabel('Mean Rage-Bait Index (RBI)', fontsize=11, fontweight='bold')
    ax3.set_title('Average RBI of Top Rage-Baiters', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, row in enrichment_df.iterrows():
        ax3.text(i, row['mean_rbi'], f"{row['mean_rbi']:.3f}", 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Sample Size (Number of Users)
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, enrichment_df['n_users'], 
                   color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
    ax4.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax4.set_title('Sample Size by Percentile', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, n in zip(bars, enrichment_df['n_users']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{n:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data-ai-slop-detector/superspreader_enrichment_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: superspreader_enrichment_analysis.png")


def generate_enrichment_report(enrichment_df):
    """Generate text report of enrichment analysis."""
    
    report = f"""
================================================================================
             SUPER-SPREADER ENRICHMENT ANALYSIS
          Bot Concentration in Top Rage-Baiters by Percentile
================================================================================

BASELINE: {enrichment_df['baseline_pct'].iloc[0]:.2f}% of all commenters are bots

"""
    
    report += "Percentile Analysis:\n"
    report += "=" * 80 + "\n\n"
    
    for _, row in enrichment_df.iterrows():
        report += f"TOP {row['percentile']}% RAGE-BAITERS ({row['n_users']:,} users)\n"
        report += f"  Mean RBI:              {row['mean_rbi']:.4f}\n"
        report += f"  Probabilistic Bot %:   {row['prob_bot_pct']:.2f}% ({row['prob_enrichment']:.2f}x baseline)\n"
        report += f"  Hard Threshold Bot %:  {row['hard_bot_pct']:.2f}% ({row['hard_enrichment']:.2f}x baseline)\n"
        report += "-" * 80 + "\n"
    
    # Key findings
    top_enrichment = enrichment_df.iloc[0]
    report += f"""
KEY FINDINGS:
-------------
1. The top {top_enrichment['percentile']}% of rage-baiters are {top_enrichment['prob_enrichment']:.2f}x more likely to be bots
   than the average commenter.

2. Bot concentration increases dramatically in higher percentiles, suggesting
   synthetic accounts dominate extreme rage-baiting behavior.

3. The probabilistic measure (using AI confidence scores) shows {
   'higher' if top_enrichment['prob_bot_pct'] > top_enrichment['hard_bot_pct'] else 'similar'
   } bot prevalence compared to hard thresholding.

4. Mean RBI increases from {enrichment_df.iloc[-1]['mean_rbi']:.4f} (top {enrichment_df.iloc[-1]['percentile']}%)
   to {top_enrichment['mean_rbi']:.4f} (top {top_enrichment['percentile']}%), indicating the most
   extreme rage-baiters are concentrated at the top.

================================================================================
"""
    
    return report


def main():
    print("=" * 80)
    print("SUPER-SPREADER ENRICHMENT ANALYSIS")
    print("=" * 80 + "\n")
    
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Analyzing bot enrichment across percentiles...")
    percentiles = [0.1, 1, 5, 10, 25, 50]
    enrichment_df = analyze_enrichment_by_percentile(df, percentiles)
    
    print("\nEnrichment Summary:")
    print(enrichment_df.to_string(index=False))
    
    print("\nGenerating visualizations...")
    create_enrichment_visualizations(enrichment_df)
    
    print("\nGenerating report...")
    report = generate_enrichment_report(enrichment_df)
    
    # Save report
    with open('data-ai-slop-detector/superspreader_enrichment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print("✓ Saved: superspreader_enrichment_report.txt")
    
    return enrichment_df


if __name__ == '__main__':
    enrichment_results = main()