
# Precomputes sentence segmentation and lemmatization for the entire EU Parliament corpus.
# Uses maximum parallelism to create sentence_df_cache.pkl for Phase 1 reuse,
# eliminating the lemmatization bottleneck in subsequent pipeline runs.

import pandas as pd
import nltk
import spacy
import pickle
import os
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')


CACHE_VERSION = 2

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt_tab', quiet=True)

def load_dataset():
    print("Loading EU Parliament dataset...")
    
    if os.path.exists("EUPDCorp_enhanced.csv"):
        print("Loading from EUPDCorp_enhanced.csv...")
        df = pd.read_csv("EUPDCorp_enhanced.csv")
    else:
        print("Loading from RDS file...")
        import pyreadr
        result = pyreadr.read_r("data/EUPDCorp_1999-2024_v1.RDS")
        df = result[None]
        df['word_count'] = df['speech_en'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    print(f"Dataset loaded: {df.shape[0]:,} rows")
    return df

def extract_valid_political_groups(df):
    all_groups = df['epg_short'].dropna().unique().tolist()
    all_groups.sort()
    print(f"Extracted {len(all_groups)} political groups")
    return all_groups

def filter_data_by_groups(df, target_groups):
    print(f"Filtering data by {len(target_groups)} political groups...")
    
    filtered_df = df[df['epg_short'].isin(target_groups)].copy()
    print(f"Filtered dataset shape: {filtered_df.shape}")
    
    filtered_df = filtered_df.dropna(subset=['speech_en', 'term_no'])
    print(f"After removing missing values: {filtered_df.shape}")
    
    return filtered_df

def process_speech_chunk(chunk):
    """Process a chunk of speeches - top-level function for multiprocessing"""
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    sent_tokenizer = nltk.sent_tokenize
    
    results = []
    for speech_info in chunk:
        sentences = sent_tokenizer(speech_info['speech'])
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                results.append({
                    'sentence': sentence,
                    'term_no': speech_info['term_no'],
                    'year': speech_info['year'],
                    'epg_short': speech_info['epg_short'],
                    'original_speech_idx': speech_info['idx']
                })
    return results

def segment_sentences_parallel(df):
    print("\nPerforming sentence segmentation...")
    
    download_nltk_data()
    
    valid_mask = df['speech_en'].notna() & (df['speech_en'].str.len() > 10)
    valid_speeches = df[valid_mask]
    print(f"Processing {len(valid_speeches):,} valid speeches")
    
    speech_data = []
    for idx, row in valid_speeches.iterrows():
        speech_data.append({
            'idx': idx,
            'speech': row['speech_en'].strip(),
            'term_no': row['term_no'],
            'year': row['year'],
            'epg_short': row['epg_short']
        })
    
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for sentence segmentation")
    
    chunk_size = max(1, len(speech_data) // (num_cores * 4))
    chunks = [speech_data[i:i + chunk_size] for i in range(0, len(speech_data), chunk_size)]
    
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} speeches each")
    
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_speech_chunk, chunks),
            total=len(chunks),
            desc="Segmenting speeches"
        ))

    all_sentences = []
    for chunk_result in results:
        all_sentences.extend(chunk_result)
    
    sentence_df = pd.DataFrame(all_sentences)
    print(f"Segmentation complete. Total sentences: {len(sentence_df):,}")
    
    return sentence_df

def lemmatize_batch(sentences_batch):
    """Lemmatize a batch of sentences - top-level function for multiprocessing"""
    import spacy

    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError:
        return [set() for _ in sentences_batch]
    
    results = []
    for sentence in sentences_batch:
        try:
            doc = nlp(sentence.lower())
            lemmas = {token.lemma_ for token in doc 
                     if token.is_alpha and len(token.lemma_) > 1}
            results.append(lemmas)
        except Exception:
            results.append(set())
    
    return results

def lemmatize_sentences_parallel(sentence_df):
    print(f"\nLemmatizing {len(sentence_df):,} sentences with maximum parallelism...")
    
    try:
        import spacy
        spacy.load('en_core_web_sm')
    except OSError:
        print("Error: SpaCy model 'en_core_web_sm' not found.")
        print("Please install it with: python -m spacy download en_core_web_sm")
        raise
    
    sentences = sentence_df['sentence'].tolist()
    
    num_cores = mp.cpu_count()
    print(f"Detected {num_cores} CPU cores")
    
    batch_size = max(50, len(sentences) // (num_cores * 8))  # 8x cores for fine-grained parallelism
    
    sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    
    print(f"Split into {len(sentence_batches)} batches of ~{batch_size} sentences each")
    print(f"Starting lemmatization with {num_cores} processes...")
    
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(lemmatize_batch, sentence_batches),
            total=len(sentence_batches),
            desc="Lemmatizing batches"
        ))
    
    all_lemmas = []
    for batch_result in results:
        all_lemmas.extend(batch_result)
    
    sentence_df['sentence_lemmas'] = all_lemmas
    
    print(f"Lemmatization complete for {len(sentence_df):,} sentences")
    return sentence_df

def save_cache(sentence_df, cache_file="sentence_df_cache.pkl"):
    print(f"\nSaving processed sentences to {cache_file}...")
    
    cache_data = {
        'version': CACHE_VERSION,
        'data': sentence_df
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✓ Cache saved to: {cache_file}")
    
    total_sentences = len(sentence_df)
    total_lemmas = sum(len(lemmas) for lemmas in sentence_df['sentence_lemmas'])
    avg_lemmas = total_lemmas / total_sentences if total_sentences > 0 else 0
    
    print(f"\nCache Statistics:")
    print(f"  Total sentences: {total_sentences:,}")
    print(f"  Total unique lemmas: {total_lemmas:,}")
    print(f"  Average lemmas per sentence: {avg_lemmas:.1f}")
    
    return cache_file

def main():
    print("=" * 80)
    print("EU PARLIAMENT SENTENCE LEMMATIZATION - MAXIMUM PARALLELISM")
    print("=" * 80)
    
    start_time = pd.Timestamp.now()
    
    try:
        if os.path.exists("sentence_df_cache.pkl"):
            print("Cache file already exists. Loading to check version...")
            
            with open("sentence_df_cache.pkl", 'rb') as f:
                cached_data = pickle.load(f)
            
            if isinstance(cached_data, dict) and cached_data.get('version') == CACHE_VERSION:
                sentence_df = cached_data['data']
                print(f"✓ Valid cache found with {len(sentence_df):,} sentences and lemmas")
                print("✓ No processing needed - cache is up to date")
                return
            else:
                print("Cache exists but is outdated - will recreate")
        
        print("\n" + "=" * 50)
        print("STEP 1: LOADING DATASET")
        print("=" * 50)
        
        df = load_dataset()
        target_groups = extract_valid_political_groups(df)
        filtered_df = filter_data_by_groups(df, target_groups)
        
        print("\n" + "=" * 50)
        print("STEP 2: SENTENCE SEGMENTATION")
        print("=" * 50)
        
        sentence_df = segment_sentences_parallel(filtered_df)
        
        print("\n" + "=" * 50)
        print("STEP 3: SENTENCE LEMMATIZATION")
        print("=" * 50)
        
        sentence_df = lemmatize_sentences_parallel(sentence_df)
        
        print("\n" + "=" * 50)
        print("STEP 4: SAVING CACHE")
        print("=" * 50)
        
        cache_path = save_cache(sentence_df)
        
        end_time = pd.Timestamp.now()
        total_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Total processing time: {total_time}")
        print(f"Cache file: {cache_path}")
        print(f"Sentences processed: {len(sentence_df):,}")
        print("This cache will be automatically used by Phase 1 for instant startup!")
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()