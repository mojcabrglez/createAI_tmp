from bertopic import BERTopic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

###
# Constants

user_cols = ["u_intro", "u_basic job description",
             "u_walkthrough", "u_project example",
             "u_dynamic", "u_changed aspects",
             "u_concerns", "u_future", "u_extra comments"]

section_order_all = [
    'basic_job_description', 'walkthrough', 'project_example',
    'dynamic', 'changed_aspects', 'concerns', 'future', 'extra_comments'
]
section_order_selected = [
    'basic_job_description', 'walkthrough', 'project_example', 'dynamic'
]

green = sns.color_palette('colorblind')[2]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jobs(filepath: str) -> pd.DataFrame:
    """Load transcript_id -> creative_type mapping from a TSV file."""
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    df['transcript_id'] = df['transcript_id'].str.strip()
    df['creative_type'] = df['creative_type'].str.strip()
    return df[['transcript_id', 'creative_type']]


def make_output_dir(base_dir: str, label: str) -> Path:
    """Create and return a subdirectory for a given creative_type label."""
    p = Path(base_dir) / label
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_section_order(df: pd.DataFrame, sections: list, col: str = 'section') -> pd.DataFrame:
    df = df[df[col].isin(sections)].copy()
    df[col] = pd.Categorical(df[col], categories=sections, ordered=True)
    return df


def calculate_ratios(group: pd.DataFrame) -> pd.Series:
    topics = group['topic_BERTopic'].values
    probs = group['topic_BERTopic_probs'].values

    count_0 = (topics == 0).sum() + 1
    count_1 = (topics == 1).sum() + 1

    weighted_0 = probs[topics == 0].sum() + 1
    weighted_1 = probs[topics == 1].sum() + 1

    return pd.Series({
        'simple_ratio': count_0 / count_1 / (count_0 + count_1),
        'weighted_ratio': weighted_0 / weighted_1 / (count_0 + count_1),
        'topic_0_count': count_0 - 1,
        'topic_1_count': count_1 - 1
    })


# ── Plotting functions (accept an output dir so they can be scoped per type) ──

def plot_average_topic_counts(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        topic_0_count=('topic_0_count', 'mean'),
        topic_1_count=('topic_1_count', 'mean')
    ).reset_index()

    section_avg_melted = section_avg.melt(
        id_vars='section',
        value_vars=['topic_0_count', 'topic_1_count'],
        var_name='Topic', value_name='Count'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Count', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section{suffix}')
    ax.set_xlabel('Section')
    ax.set_ylabel('Average Count')
    ax.tick_params(axis='x', rotation=45)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['USER agency', 'AI use'], title='Topic')
    plt.tight_layout()
    plt.savefig(out_dir / f'average_ratio_count_selected{suffix}.png')
    plt.close()


def plot_simple_ratio(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        simple_ratio=('simple_ratio', 'mean')
    ).reset_index()
    section_avg_melted = section_avg.melt(
        id_vars='section', value_vars=['simple_ratio'],
        var_name='Topic', value_name='Ratio'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Ratio', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section{suffix}')
    ax.set_xlabel('Section')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Average Ratio, 1 = USER agency, 0 = AI use')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / f'average_ratio_simple_selected{suffix}.png')
    plt.close()


def plot_weighted_ratio(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        weighted_ratio=('weighted_ratio', 'mean')
    ).reset_index()
    section_avg_melted = section_avg.melt(
        id_vars='section', value_vars=['weighted_ratio'],
        var_name='Topic', value_name='Ratio'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Ratio', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section, weighed by probability{suffix}')
    ax.set_xlabel('Section')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Average Ratio, 1 = USER agency, 0 = AI use')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / f'average_ratio_weighed_selected{suffix}.png')
    plt.close()


def compute_section_similarities(df: pd.DataFrame, sections: list) -> pd.DataFrame:
    """Compute per-section average similarities and derived metrics."""
    sub = apply_section_order(df, sections)
    sims = sub.groupby('section', observed=False).agg(
        sim_to_HUMAN=('sim_to_HUMAN', 'mean'),
        sim_to_AI=('sim_to_AI', 'mean')
    ).reset_index()

    sims['sim_difference'] = sims['sim_to_HUMAN'] - sims['sim_to_AI']

    min_d, max_d = sims['sim_difference'].min(), sims['sim_difference'].max()
    denom = max_d - min_d
    sims['normalized_difference'] = (
        (sims['sim_difference'] - min_d) / denom if denom != 0 else 0.5
    )
    sims['balanced_ratio'] = (
        (sims['sim_to_HUMAN'] - sims['sim_to_AI']) /
        (sims['sim_to_HUMAN'] + sims['sim_to_AI'])
    )
    return sims


def plot_similarity_balance_normalized(section_sims: pd.DataFrame, out_dir: Path,
                                       suffix: str = "", filename: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    section_sims_melted = section_sims.melt(
        id_vars='section',
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    ax1 = axes[0]
    sns.barplot(data=section_sims_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 0.6)
    ax1.set_title(f'Average Cosine Similarity by Section{suffix}')
    ax1.set_xlabel('Section')
    ax1.set_ylabel('Average Similarity')
    ax1.tick_params(axis='x', rotation=45)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['USER Vector', 'AI USE Vector'],
               title='Reference Vector', loc="upper right")

    ax2 = axes[1]
    neutral = 0.5
    bar_lengths = section_sims['normalized_difference'].values - neutral
    colors = [green if v > neutral else 'skyblue'
              for v in section_sims['normalized_difference']]
    ax2.barh(section_sims['section'], bar_lengths, left=neutral,
             color=colors, alpha=0.7)
    ax2.axvline(x=0.5, color='grey', linestyle='--', linewidth=2, label='Neutral')
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title(f'Similarity Balance: 0 = AI, 1 = USER{suffix}')
    ax2.set_ylabel('Section')
    ax2.set_xlabel('Normalized Difference')
    ax2.legend(loc='lower right')
    ax2.invert_yaxis()

    plt.tight_layout()
    fname = filename or f'similarity_balance_normalized{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def plot_boxplots_both_similarities(df: pd.DataFrame, sections: list,
                                    out_dir: Path, suffix: str = "", filename: str = ""):
    sub = apply_section_order(df, sections)
    sub['balanced_ratio'] = (
        (sub['sim_to_HUMAN'] - sub['sim_to_AI']) /
        (sub['sim_to_HUMAN'] + sub['sim_to_AI'])
    )

    triplets_melted = sub.melt(
        id_vars=['section', 'transcript_id'],
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.boxplot(data=triplets_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f'Distribution of Similarities by Section{suffix}')
    ax1.set_xlabel('Section')
    ax1.set_ylabel('Cosine Similarity')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['USER Agency', 'AI Use'],
               title='Reference Vector', loc='upper right')

    ax2 = axes[1]
    sns.boxplot(data=sub, x='section', y='balanced_ratio', ax=ax2, color=green)
    ax2.axhline(y=0, color='grey', linestyle=':', linewidth=2, label='Equal similarity')
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title(f'Balanced Similarity Ratio by Section\n(-1=AI, +1=USER){suffix}')
    ax2.set_xlabel('Section')
    ax2.set_ylabel('Balanced Ratio')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    fname = filename or f'similarity_balanced_ratio{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def plot_normalized_difference_boxplots(df: pd.DataFrame, sections: list,
                                        out_dir: Path, suffix: str = "", filename: str = ""):
    sub = apply_section_order(df, sections).copy()
    sub['sim_difference_individual'] = sub['sim_to_HUMAN'] - sub['sim_to_AI']
    min_d, max_d = sub['sim_difference_individual'].min(), sub['sim_difference_individual'].max()
    denom = max_d - min_d
    sub['normalized_difference'] = (
        (sub['sim_difference_individual'] - min_d) / denom if denom != 0 else 0.5
    )

    triplets_melted = sub.melt(
        id_vars=['section', 'transcript_id'],
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.boxplot(data=triplets_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f'Distribution of Similarities by Section{suffix}')
    ax1.set_xlabel('Section')
    ax1.set_ylabel('Cosine Similarity')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['USER Agency', 'AI Use'],
               title='Reference Vector', loc='upper right')

    ax2 = axes[1]
    sns.boxplot(data=sub, x='section', y='normalized_difference', ax=ax2, color=green)
    ax2.axhline(y=0.5, color='grey', linestyle=':', linewidth=2, label='Equal similarity')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title(f'Normalized Similarity Difference by Section\n(0=AI, 1=USER){suffix}')
    ax2.set_xlabel('Section')
    ax2.set_ylabel('Normalized Difference')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    fname = filename or f'similarity_normalized_differences{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def run_analysis_for_group(df: pd.DataFrame, out_dir: Path,
                           label: str = "", ratio_df: pd.DataFrame = None):
    """
    Run the full plotting + summary pipeline for one subset of all_triplets.
    df       : rows of all_triplets (already has sim_to_HUMAN, sim_to_AI, etc.)
    out_dir  : where to write outputs
    label    : display label for plot titles (e.g. '' or ' [writer]')
    ratio_df : pre-computed BERTopic ratio dataframe for the same subset
    """
    suffix = f" [{label}]" if label else ""

    # ── BERTopic ratio plots (only when ratio_df is provided) ─────────────────
    if ratio_df is not None:
        ratio_sub = apply_section_order(ratio_df, section_order_selected)
        plot_average_topic_counts(ratio_sub, out_dir, suffix)
        plot_simple_ratio(ratio_sub, out_dir, suffix)
        plot_weighted_ratio(ratio_sub, out_dir, suffix)

    # ── ALL sections – average bar + normalized balance ────────────────────────
    section_sims_all = compute_section_similarities(df, section_order_all)
    section_sims_all.to_csv(
        out_dir / f'{label}__average_similarities_ratio_HUMAN-AI_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.tsv',
        index=False
    )
    plot_similarity_balance_normalized(
        section_sims_all, out_dir, suffix,
        filename=f'similarity_balance_normalized_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )

    # boxplots – all sections
    plot_boxplots_both_similarities(
        df, section_order_all, out_dir, suffix,
        filename=f'{label}__similarity_balanced_ratio_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )
    plot_normalized_difference_boxplots(
        df, section_order_all, out_dir, suffix,
        filename=f'{label}__similarity_normalized_differences_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )

    # ── SELECTED sections ──────────────────────────────────────────────────────
    section_sims_sel = compute_section_similarities(df, section_order_selected)

    plot_similarity_balance_normalized(
        section_sims_sel, out_dir, suffix,
        filename=f'{label}__similarity_normalized_difference_start_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )
    plot_boxplots_both_similarities(
        df, section_order_selected, out_dir, suffix,
        filename=f'{label}__similarity_balanced_difference_start_sections_all_docs{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )
    plot_normalized_difference_boxplots(
        df, section_order_selected, out_dir, suffix,
        filename=f'{label}__similarity_normalized_differences_four_sections_all_documents{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )

    # ── Summary printout ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Summary by Section [{label or 'ALL'}]:")
    print(section_sims_all[['section', 'sim_to_HUMAN', 'sim_to_AI', 'normalized_difference']])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
# ── 1. BERTopic model training (on all data, unchanged) ──────────────────────

    df = pd.read_csv('interview_split.tsv', sep='\t', index_col=0)
    df['all_text'] = df[user_cols].fillna('').astype(str).agg(''.join, axis=1)

    topic_model = BERTopic.load("./BERTriplet_model")


    all_triplets = pd.read_csv('stanza_out/all_triplets_BERTopic.tsv', sep='\t')

    # ── 3. Compute cosine similarities to topic centres (once) ───────────────────

    topic_centers = topic_model.topic_embeddings_[1:]  # skip outlier topic -1

    embeddings = topic_model.embedding_model.embedding_model.encode(
        all_triplets['full_text'].tolist()
    )
    similarities = cosine_similarity(embeddings, topic_centers)

    all_triplets['sim_to_HUMAN'] = similarities[:, 0]
    all_triplets['sim_to_AI'] = similarities[:, 1]
    all_triplets['sim_difference_individual'] = (
        all_triplets['sim_to_HUMAN'] - all_triplets['sim_to_AI']
    )

    # Normalize once across the full dataset
    min_d = all_triplets['sim_difference_individual'].min()
    max_d = all_triplets['sim_difference_individual'].max()
    all_triplets['normalized_difference'] = (
        (all_triplets['sim_difference_individual'] - min_d) / (max_d - min_d)
    )
    all_triplets['balanced_ratio'] = (
        (all_triplets['sim_to_HUMAN'] - all_triplets['sim_to_AI']) /
        (all_triplets['sim_to_HUMAN'] + all_triplets['sim_to_AI'])
    )

    # ── 4. BERTopic ratio_df (once on full data) ──────────────────────────────────

    ratio_df_full = all_triplets.groupby(
        ['transcript_id', 'section']
    ).apply(calculate_ratios).reset_index()

    ratio_df_full['section'] = pd.Categorical(
        ratio_df_full['section'], categories=section_order_all, ordered=True
    )

    # ── 5. Load creative_type mapping ─────────────────────────────────────────────

    JOBS_FILE = "interviews_by_job.tsv"   # <-- adjust path if needed
    OUTPUT_ROOT = Path("results")

    jobs_df = load_jobs(JOBS_FILE)
    creative_types = sorted(jobs_df['creative_type'].unique().tolist())
    print(f"\nFound creative types: {creative_types}")

    # ── 6. Run analysis: ALL data first ───────────────────────────────────────────

    all_out = make_output_dir(OUTPUT_ROOT, "all")
    run_analysis_for_group(all_triplets, all_out, label="", ratio_df=ratio_df_full)

    # ── 7. Run analysis: per creative_type ────────────────────────────────────────

    # Join creative_type onto all_triplets
    all_triplets_typed = all_triplets.merge(jobs_df, on='transcript_id', how='left')

    unmatched = all_triplets_typed['creative_type'].isna().sum()
    if unmatched:
        print(f"\nWARNING: {unmatched} rows have no matching creative_type → written to 'unmatched' folder.")

    for ct, group_df in all_triplets_typed.groupby('creative_type', dropna=False):
        ct_label = str(ct) if pd.notna(ct) else "unmatched"
        print(f"\n{'─' * 60}")
        print(f"Processing creative_type: {ct_label}  ({len(group_df)} rows)")

        ct_out = make_output_dir(OUTPUT_ROOT, ct_label)

        # Subset ratio_df to transcripts in this group
        ct_transcript_ids = set(group_df['transcript_id'].unique())
        ratio_df_ct = ratio_df_full[
            ratio_df_full['transcript_id'].isin(ct_transcript_ids)
        ].copy()

        run_analysis_for_group(group_df, ct_out, label=ct_label, ratio_df=ratio_df_ct)

    print(f"\n{'═' * 60}")
    print(f"Done. Results written under: {OUTPUT_ROOT}/")
    print(f"{'═' * 60}")