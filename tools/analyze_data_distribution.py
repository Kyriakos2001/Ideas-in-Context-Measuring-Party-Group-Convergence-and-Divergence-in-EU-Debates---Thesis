#!/usr/bin/env python3
# Data distribution analysis tool for EU Parliament discourse analysis.
# Analyzes speech and sentence distribution across parliamentary terms and political groups,
# identifies data imbalances, and provides statistics for stratified sampling and threshold setting.

import sys
import os
import argparse
import pandas as pd
import nltk
from collections import defaultdict, Counter
from datetime import datetime
import warnings


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from poc_phase1_data_prep import (
    load_raw_data, 
    extract_valid_political_groups,
    segment_sentences
)

warnings.filterwarnings('ignore')

def setup_output_directory():
    output_dir = "results/data_distribution"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def count_speeches_by_group(df, temporal_unit='term'):

    print(f"Counting speeches by {temporal_unit} and political group...")
    

    valid_groups = extract_valid_political_groups(df)

    filtered_df = df[df['epg_short'].isin(valid_groups)].copy()
    filtered_df = filtered_df.dropna(subset=['speech_en', 'term_no'])

    temporal_col = 'year' if temporal_unit == 'year' else 'term_no'
    speech_counts = filtered_df.groupby([temporal_col, 'epg_short']).size().reset_index(name='speech_count')
    speech_counts = speech_counts.rename(columns={temporal_col: 'temporal_unit'})
    
    print(f"Found {len(speech_counts)} combinations of ({temporal_unit}, group)")
    return speech_counts, filtered_df

def count_sentences_by_group(df, temporal_unit='term'):

    print(f"Counting sentences by {temporal_unit} and political group...")
    print("Using identical sentence segmentation method from pipeline...")

    sentence_df = segment_sentences(df)

    temporal_col = 'year' if temporal_unit == 'year' else 'term_no'
    sentence_counts = sentence_df.groupby([temporal_col, 'epg_short']).size().reset_index(name='sentence_count')
    sentence_counts = sentence_counts.rename(columns={temporal_col: 'temporal_unit'})
    
    print(f"Total sentences processed: {len(sentence_df):,}")
    return sentence_counts, sentence_df

def count_keyword_sentences(sentence_df, keyword, temporal_unit='term'):

    print(f"Counting sentences containing '{keyword}'...")
    
    import re
    keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

    contains_keyword = sentence_df['sentence'].apply(
        lambda x: bool(keyword_pattern.search(x)) if pd.notnull(x) else False
    )
    
    keyword_sentences = sentence_df[contains_keyword]
    print(f"Found {len(keyword_sentences):,} sentences containing '{keyword}'")

    temporal_col = 'year' if temporal_unit == 'year' else 'term_no'
    keyword_counts = keyword_sentences.groupby([temporal_col, 'epg_short']).size().reset_index(name='keyword_count')
    keyword_counts = keyword_counts.rename(columns={temporal_col: 'temporal_unit'})
    
    return keyword_counts

def generate_summary_statistics(combined_df, temporal_unit='term'):
    print("Generating summary statistics...")
    
    stats = {
        'total_speeches': combined_df['speech_count'].sum(),
        'total_sentences': combined_df['sentence_count'].sum(),
        'avg_sentences_per_speech': combined_df['sentence_count'].sum() / combined_df['speech_count'].sum(),
        'temporal_units': len(combined_df['temporal_unit'].unique()),
        'political_groups': len(combined_df['epg_short'].unique()),
        'combinations': len(combined_df)
    }

    temporal_stats = []
    for temporal_val in sorted(combined_df['temporal_unit'].unique()):
        temporal_data = combined_df[combined_df['temporal_unit'] == temporal_val]
        
        min_speeches = temporal_data['speech_count'].min()
        min_speech_group = temporal_data[temporal_data['speech_count'] == min_speeches]['epg_short'].iloc[0]
        
        min_sentences = temporal_data['sentence_count'].min()
        min_sentence_group = temporal_data[temporal_data['sentence_count'] == min_sentences]['epg_short'].iloc[0]
        
        temporal_stats.append({
            'temporal_unit': temporal_val,
            'min_speeches': min_speeches,
            'min_speech_group': min_speech_group,
            'min_sentences': min_sentences,
            'min_sentence_group': min_sentence_group,
            'max_speeches': temporal_data['speech_count'].max(),
            'max_sentences': temporal_data['sentence_count'].max(),
            'groups_count': len(temporal_data)
        })
    
    return stats, temporal_stats

def print_console_report(combined_df, stats, temporal_stats, keyword_df=None, keyword=None, temporal_unit='term'):

    temporal_desc = temporal_unit.title()
    
    print("\n" + "=" * 80)
    print("DATA DISTRIBUTION ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temporal Unit: {temporal_desc}")
    print()
    
    print("DATASET SUMMARY:")
    print(f"  Total Speeches: {stats['total_speeches']:,}")
    print(f"  Total Sentences: {stats['total_sentences']:,}")
    print(f"  Average Sentences/Speech: {stats['avg_sentences_per_speech']:.1f}")
    print(f"  {temporal_desc}s: {stats['temporal_units']}")
    print(f"  Political Groups: {stats['political_groups']}")
    print(f"  ({temporal_desc}, Group) Combinations: {stats['combinations']}")
    print()
    
    print(f"SPEECHES AND SENTENCES BY {temporal_desc.upper()} AND POLITICAL GROUP:")
    print("-" * 80)
    print(f"{'Term':<4} | {'Group':<12} | {'Speeches':<9} | {'Sentences':<10} | {'Avg S/S':<8}")
    print("-" * 80)
    
    for _, row in combined_df.sort_values(['temporal_unit', 'epg_short']).iterrows():
        avg_sent = row['sentence_count'] / row['speech_count'] if row['speech_count'] > 0 else 0
        print(f"{int(row['temporal_unit']):<4} | {row['epg_short']:<12} | {row['speech_count']:<9,} | {row['sentence_count']:<10,} | {avg_sent:<8.1f}")
    
    print()
    print(f"MINIMUM COUNTS PER {temporal_desc.upper()} (Critical for Stratified Sampling):")
    print("-" * 80)
    print(f"{'Term':<4} | {'Min Speeches':<12} | {'Min Group':<12} | {'Min Sentences':<13} | {'Min Group':<12}")
    print("-" * 80)
    
    for stat in temporal_stats:
        print(f"{int(stat['temporal_unit']):<4} | {stat['min_speeches']:<12,} | {stat['min_speech_group']:<12} | {stat['min_sentences']:<13,} | {stat['min_sentence_group']:<12}")
    
    if keyword_df is not None and keyword is not None:
        print()
        print(f"KEYWORD ANALYSIS ('{keyword.upper()}'):")
        print("-" * 80)
        print(f"{'Term':<4} | {'Group':<12} | {f'{keyword.title()} Sentences':<18} | {'% of Total':<12}")
        print("-" * 80)
        
        # Merge with combined_df to get percentages
        keyword_merged = combined_df.merge(keyword_df, on=['temporal_unit', 'epg_short'], how='left')
        keyword_merged['keyword_count'] = keyword_merged['keyword_count'].fillna(0)
        keyword_merged['percentage'] = (keyword_merged['keyword_count'] / keyword_merged['sentence_count'] * 100).fillna(0)
        
        for _, row in keyword_merged.sort_values(['temporal_unit', 'epg_short']).iterrows():
            if row['keyword_count'] > 0:
                print(f"{int(row['temporal_unit']):<4} | {row['epg_short']:<12} | {int(row['keyword_count']):<18,} | {row['percentage']:<12.1f}%")

def save_detailed_reports(combined_df, stats, temporal_stats, output_dir, keyword_df=None, keyword=None, temporal_unit='term'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_filename = f"data_distribution_report_{temporal_unit}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    if keyword_df is not None:
        detailed_df = combined_df.merge(keyword_df, on=['temporal_unit', 'epg_short'], how='left')
        detailed_df['keyword_count'] = detailed_df['keyword_count'].fillna(0)
        detailed_df[f'{keyword}_percentage'] = (detailed_df['keyword_count'] / detailed_df['sentence_count'] * 100).fillna(0)
    else:
        detailed_df = combined_df.copy()
    
    detailed_df['avg_sentences_per_speech'] = detailed_df['sentence_count'] / detailed_df['speech_count']
    detailed_df.to_csv(csv_path, index=False)
    print(f"Detailed CSV report saved to: {csv_path}")
    
    # Save text summary
    txt_filename = f"data_distribution_summary_{temporal_unit}_{timestamp}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        f.write(f"EU Parliament Data Distribution Analysis\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Temporal Unit: {temporal_unit.title()}\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write(f"  Total Speeches: {stats['total_speeches']:,}\n")
        f.write(f"  Total Sentences: {stats['total_sentences']:,}\n")
        f.write(f"  Average Sentences/Speech: {stats['avg_sentences_per_speech']:.1f}\n")
        f.write(f"  {temporal_unit.title()}s: {stats['temporal_units']}\n")
        f.write(f"  Political Groups: {stats['political_groups']}\n")
        f.write(f"  Combinations: {stats['combinations']}\n\n")
        
        f.write(f"MINIMUM COUNTS PER {temporal_unit.upper()} (for Stratified Sampling):\n")
        for stat in temporal_stats:
            f.write(f"  {temporal_unit.title()} {int(stat['temporal_unit'])}:\n")
            f.write(f"    Min Speeches: {stat['min_speeches']:,} ({stat['min_speech_group']})\n")
            f.write(f"    Min Sentences: {stat['min_sentences']:,} ({stat['min_sentence_group']})\n")
            f.write(f"    Max Speeches: {stat['max_speeches']:,}\n")
            f.write(f"    Max Sentences: {stat['max_sentences']:,}\n")
            f.write(f"    Groups: {stat['groups_count']}\n\n")
        
        if keyword and keyword_df is not None:
            total_keyword_sentences = detailed_df['keyword_count'].sum()
            f.write(f"KEYWORD ANALYSIS ('{keyword}'):\n")
            f.write(f"  Total {keyword} sentences: {int(total_keyword_sentences):,}\n")
            f.write(f"  Percentage of all sentences: {total_keyword_sentences/stats['total_sentences']*100:.2f}%\n\n")
    
    print(f"Summary report saved to: {txt_path}")
    
    return csv_path, txt_path

def main():

    parser = argparse.ArgumentParser(description='Analyze EU Parliament data distribution')
    parser.add_argument('--keyword', default='security', help='Keyword to analyze (default: security)')
    parser.add_argument('--temporal-unit', default='term', choices=['term', 'year'], 
                       help='Temporal unit for analysis (default: term)')
    
    args = parser.parse_args()
    
    print("EU Parliament Data Distribution Analysis")
    print("=" * 50)
    print(f"Keyword: {args.keyword}")
    print(f"Temporal Unit: {args.temporal_unit}")
    print()
    

    output_dir = setup_output_directory()
    

    try:
        print("Loading dataset...")
        try:
            df = load_raw_data("results")
        except:
            df = load_raw_data(".")
        
        print(f"Loaded dataset with {len(df):,} rows")
        print()

        speech_counts, filtered_df = count_speeches_by_group(df, args.temporal_unit)
        print()

        sentence_counts, sentence_df = count_sentences_by_group(filtered_df, args.temporal_unit)
        print()

        keyword_counts = count_keyword_sentences(sentence_df, args.keyword, args.temporal_unit)
        print()

        combined_df = speech_counts.merge(sentence_counts, on=['temporal_unit', 'epg_short'], how='outer')
        combined_df = combined_df.fillna(0)

        stats, temporal_stats = generate_summary_statistics(combined_df, args.temporal_unit)

        print_console_report(combined_df, stats, temporal_stats, keyword_counts, args.keyword, args.temporal_unit)

        print("\nSaving reports...")
        csv_path, txt_path = save_detailed_reports(
            combined_df, stats, temporal_stats, output_dir, 
            keyword_counts, args.keyword, args.temporal_unit
        )
        
        print()
        print("Analysis complete!")
        print(f"Reports saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()