import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import pyreadr
import re
import nltk
from collections import Counter

warnings.filterwarnings('ignore')


def load_and_inspect_data(file_path):

    print("=" * 60)
    print("1. INITIAL DATA LOADING AND INSPECTION")
    print("=" * 60)


    print("Loading RDS data...")
    result = pyreadr.read_r(file_path)
    df = result[None]


    print(f"\nDataset shape: {df.shape}")
    print(f"Total speeches: {df.shape[0]:,}")
    print(f"Total variables: {df.shape[1]}")

    print("\nColumn information:")
    print(df.info())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))


    critical_cols = ['speech_en', 'epg_short', 'term_no']
    print(f"\nCritical columns status:")
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"  {col}: {missing:,} missing ({missing / len(df) * 100:.2f}%)")
        else:
            print(f"  {col}: Column not found!")

    return df


def analyze_speech_distribution(df):

    print("\n" + "=" * 60)
    print("2. SPEECH DISTRIBUTION ANALYSIS")
    print("=" * 60)


    print("\nSpeeches per Political Group:")
    group_counts = df['epg_short'].value_counts()
    print(group_counts)


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))


    group_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Number of Speeches per Political Group')
    axes[0, 0].set_xlabel('Political Group')
    axes[0, 0].set_ylabel('Number of Speeches')
    axes[0, 0].tick_params(axis='x', rotation=45)


    print(f"\nSpeeches per Parliamentary Term:")
    term_counts = df['term_no'].value_counts().sort_index()
    print(term_counts)


    term_counts.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Number of Speeches per Parliamentary Term')
    axes[0, 1].set_xlabel('Parliamentary Term')
    axes[0, 1].set_ylabel('Number of Speeches')


    group_term_crosstab = pd.crosstab(df['term_no'], df['epg_short'])
    group_term_crosstab.plot(kind='bar', stacked=True, ax=axes[1, 0], figsize=(10, 6))
    axes[1, 0].set_title('Political Group Distribution Over Parliamentary Terms')
    axes[1, 0].set_xlabel('Parliamentary Term')
    axes[1, 0].set_ylabel('Number of Speeches')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    sns.heatmap(group_term_crosstab.T, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Heatmap: Political Groups vs Parliamentary Terms')
    axes[1, 1].set_xlabel('Parliamentary Term')
    axes[1, 1].set_ylabel('Political Group')

    plt.tight_layout()
    plt.savefig('speech_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


    print(f"\nDistribution Summary:")
    print(f"Most active political group: {group_counts.index[0]} ({group_counts.iloc[0]:,} speeches)")
    print(f"Least active political group: {group_counts.index[-1]} ({group_counts.iloc[-1]:,} speeches)")
    print(f"Term with most speeches: Term {term_counts.index[-1]} ({term_counts.iloc[-1]:,} speeches)")
    print(f"Term with least speeches: Term {term_counts.index[0]} ({term_counts.iloc[0]:,} speeches)")


def analyze_speech_characteristics(df):

    print("\n" + "=" * 60)
    print("3. SPEECH CHARACTERISTICS ANALYSIS")
    print("=" * 60)


    print("Calculating word counts for each speech...")
    df['word_count'] = df['speech_en'].fillna('').astype(str).apply(lambda x: len(x.split()))

    print(f"\nSpeech Length Statistics:")
    print(df['word_count'].describe())


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))


    axes[0, 0].hist(df['word_count'], bins=50, color='lightblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Speech Lengths (Word Count)')
    axes[0, 0].set_xlabel('Word Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['word_count'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["word_count"].mean():.0f}')
    axes[0, 0].legend()


    axes[0, 1].hist(df['word_count'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Distribution of Speech Lengths (Log Scale)')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency (Log Scale)')


    df.boxplot(column='word_count', by='epg_short', ax=axes[1, 0])
    axes[1, 0].set_title('Speech Length Distribution by Political Group')
    axes[1, 0].set_xlabel('Political Group')
    axes[1, 0].set_ylabel('Word Count')
    axes[1, 0].tick_params(axis='x', rotation=45)


    df.boxplot(column='word_count', by='term_no', ax=axes[1, 1])
    axes[1, 1].set_title('Speech Length Distribution by Parliamentary Term')
    axes[1, 1].set_xlabel('Parliamentary Term')
    axes[1, 1].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig('speech_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


    print(f"\nAverage speech length by political group:")
    group_stats = df.groupby('epg_short')['word_count'].agg(['mean', 'std', 'median']).round(2)
    print(group_stats.sort_values('mean', ascending=False))


    short_threshold = df['word_count'].quantile(0.05)
    long_threshold = df['word_count'].quantile(0.95)

    print(f"\nSpeech Length Categories:")
    print(
        f"Very short speeches (<{short_threshold:.0f} words): {(df['word_count'] < short_threshold).sum():,} ({(df['word_count'] < short_threshold).mean() * 100:.1f}%)")
    print(
        f"Very long speeches (>{long_threshold:.0f} words): {(df['word_count'] > long_threshold).sum():,} ({(df['word_count'] > long_threshold).mean() * 100:.1f}%)")

    return df


def preliminary_keyword_analysis(df, keywords=None):

    print("\n" + "=" * 60)
    print("4. PRELIMINARY KEYWORD ANALYSIS (KEYWORD SCOUTING)")
    print("=" * 60)

    if keywords is None:
        keywords = ['security', 'integration', 'sovereignty', 'democracy', 'economy',
                    'migration', 'climate', 'trade', 'cooperation', 'unity']

    print(f"Analyzing keywords: {keywords}")


    speeches_lower = df['speech_en'].fillna('').astype(str).str.lower()


    keyword_stats = {}
    for keyword in keywords:

        contains_keyword = speeches_lower.str.contains(keyword, na=False)
        doc_count = contains_keyword.sum()
        doc_percentage = (doc_count / len(df)) * 100


        total_mentions = speeches_lower.str.count(keyword).sum()

        keyword_stats[keyword] = {
            'documents_containing': doc_count,
            'document_percentage': doc_percentage,
            'total_mentions': total_mentions
        }


    keyword_df = pd.DataFrame(keyword_stats).T
    keyword_df = keyword_df.sort_values('documents_containing', ascending=False)

    print(f"\nKeyword Frequency Summary:")
    print(keyword_df)


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))


    keyword_df['documents_containing'].plot(kind='bar', ax=axes[0, 0], color='coral')
    axes[0, 0].set_title('Number of Documents Containing Each Keyword')
    axes[0, 0].set_xlabel('Keywords')
    axes[0, 0].set_ylabel('Number of Documents')
    axes[0, 0].tick_params(axis='x', rotation=45)


    keyword_df['document_percentage'].plot(kind='bar', ax=axes[0, 1], color='lightblue')
    axes[0, 1].set_title('Percentage of Documents Containing Each Keyword')
    axes[0, 1].set_xlabel('Keywords')
    axes[0, 1].set_ylabel('Percentage of Documents')
    axes[0, 1].tick_params(axis='x', rotation=45)


    group_keyword_data = []
    for group in df['epg_short'].unique():
        if pd.isna(group):
            continue
        group_speeches = df[df['epg_short'] == group]['speech_en'].fillna('').astype(str).str.lower()
        group_data = {'Political_Group': group}

        for keyword in keywords[:5]:
            contains_keyword = group_speeches.str.contains(keyword, na=False)
            percentage = (contains_keyword.sum() / len(group_speeches)) * 100
            group_data[keyword] = percentage

        group_keyword_data.append(group_data)

    group_keyword_df = pd.DataFrame(group_keyword_data)
    group_keyword_df = group_keyword_df.set_index('Political_Group')


    sns.heatmap(group_keyword_df.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title('Keyword Usage by Political Group (% of Documents)')
    axes[1, 0].set_xlabel('Political Group')
    axes[1, 0].set_ylabel('Keywords')


    top_keywords = keyword_df.head(8)
    axes[1, 1].pie(top_keywords['documents_containing'], labels=top_keywords.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Distribution of Top 8 Keywords by Document Count')

    plt.tight_layout()
    plt.savefig('keyword_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


    most_common = keyword_df.index[0]
    least_common = keyword_df.index[-1]

    print(f"\nKeyword Analysis Summary:")
    print(
        f"Most frequently mentioned keyword: '{most_common}' (in {keyword_df.loc[most_common, 'documents_containing']:,} documents, {keyword_df.loc[most_common, 'document_percentage']:.1f}%)")
    print(
        f"Least frequently mentioned keyword: '{least_common}' (in {keyword_df.loc[least_common, 'documents_containing']:,} documents, {keyword_df.loc[least_common, 'document_percentage']:.1f}%)")


    print(f"\nData Sufficiency Check:")
    sufficient_keywords = keyword_df[keyword_df['documents_containing'] >= 100]
    print(f"Keywords with ≥100 mentions suitable for detailed analysis: {len(sufficient_keywords)}")
    if len(sufficient_keywords) > 0:
        print(f"Suitable keywords: {list(sufficient_keywords.index)}")

    return keyword_df


def analyze_top_words(df, n_top=500):

    print("\n" + "=" * 60)
    print(f"5. TOP {n_top} MOST FREQUENT WORDS ANALYSIS (EXCLUDING STOPWORDS)")
    print("=" * 60)

    try:
        from nltk.corpus import stopwords
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        print(f"Loaded {len(stop_words)} English stopwords from NLTK.")
    except ImportError:
        print("NLTK stopwords not found. Please run 'pip install nltk' and try again.")
        return
    except Exception as e:
        print(f"Error loading NLTK stopwords: {e}. Skipping top words analysis.")
        return


    print("Combining all speech text for analysis...")

    all_speeches = df['speech_en'].dropna().astype(str)
    all_text_lower = " ".join(all_speeches).lower()


    print("Tokenizing and counting all words (this may take a moment)...")
    words = re.findall(r'\b[a-z]{2,}\b', all_text_lower)
    total_word_count = len(words)
    print(f"Total words found (raw tokens): {total_word_count:,}")


    non_stop_words = [word for word in words if word not in stop_words]
    non_stop_word_count = len(non_stop_words)

    print(f"Total words after filtering stopwords: {non_stop_word_count:,}")
    print(f"({(total_word_count - non_stop_word_count):,} stopwords removed)")


    word_counts = Counter(non_stop_words)
    most_common_words = word_counts.most_common(n_top)


    print(f"\nTop {n_top} Most Frequent Words (Non-Stopwords):")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Word':<20} | {'Frequency':<10}")
    print("-" * 60)

    output_lines = []
    for i, (word, count) in enumerate(most_common_words, 1):
        line = f"{i:<5} | {word:<20} | {count:<10,}"
        print(line)
        output_lines.append(f"{i}. {word} ({count:,})")


    try:
        report_path = 'top_100_words_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Top 100 Most Frequent Words (Non-Stopwords) in Corpus\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Raw Tokens Analyzed: {total_word_count:,}\n")
            f.write(f"Total Tokens (No Stopwords): {non_stop_word_count:,}\n")
            f.write("=" * 60 + "\n")
            f.write("\n".join(output_lines))
        print(f"\nSummary report saved to: {report_path}")
    except Exception as e:
        print(f"\nFailed to save summary report: {e}")

    return most_common_words


def main():

    file_path = 'data/EUPDCorp_1999-2024_v1.RDS'

    try:

        df = load_and_inspect_data(file_path)


        analyze_speech_distribution(df)


        df = analyze_speech_characteristics(df)


        keyword_results = preliminary_keyword_analysis(df)


        analyze_top_words(df)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("- speech_distribution_analysis.png")
        print("- speech_characteristics_analysis.png")
        print("- keyword_analysis.png")
        print("- top_100_words_report.txt")


        df.to_csv('EUPDCorp_enhanced.csv', index=False)
        print("- EUPDCorp_enhanced.csv (with word_count column)")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please make sure the file is in the current directory or update the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()