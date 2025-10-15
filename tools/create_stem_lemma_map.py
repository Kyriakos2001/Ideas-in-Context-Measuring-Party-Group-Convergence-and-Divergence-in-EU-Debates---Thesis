"""
Create Stem-to-Lemma Mapping Script

a one-time analysis of the entire EU Parliament speech corpus
Creates a stem-to-lemma mapping dictionary. This enables efficient morphological
variation detection for keyword analysis.

Input: EUPDCorp_enhanced.csv or RDS fallback
Output: stem_to_lemma_map.pkl

Example output structure:
{
  "immigr": {"immigrant", "immigration", "immigrate"},
  "secur":  {"security", "secure"},
  "polici": {"policy"},
  ...
}
"""

import pandas as pd
import nltk
import spacy
import re
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

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
        print("CSV not found, loading from RDS file...")
        import pyreadr
        result = pyreadr.read_r("data/EUPDCorp_1999-2024_v1.RDS")
        df = result[None]
    
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def extract_all_text(df):
    print("Extracting all speech text...")
    
    valid_speeches = df['speech_en'].dropna()
    print(f"Found {len(valid_speeches):,} speeches with text")
    
    print("Combining all text (this may take a moment)...")
    all_text = ' '.join(valid_speeches.astype(str)).lower()
    
    print(f"Total text length: {len(all_text):,} characters")
    return all_text

def extract_vocabulary(all_text):
    print("Extracting vocabulary...")
    
    print("Finding all words with regex...")
    words = re.findall(r'\b[a-z]{2,}\b', all_text)
    
    print(f"Total word occurrences: {len(words):,}")
    
    vocabulary = set(words)
    print(f"Unique vocabulary size: {len(vocabulary):,} words")
    
    return vocabulary

def initialize_nlp_tools():
    print("Initializing NLP tools...")
    
    download_nltk_data()
    
    stemmer = nltk.stem.SnowballStemmer('english')
    print("NLTK Snowball Stemmer initialized")
    
    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        print("SpaCy lemmatizer initialized (parser and NER disabled for speed)")
    except OSError:
        print("Error: SpaCy model 'en_core_web_sm' not found.")
        print("Please install it with: python -m spacy download en_core_web_sm")
        raise
    
    return stemmer, nlp

def build_stem_lemma_map(vocabulary, stemmer, nlp):
    print("Building stem-to-lemma mapping...")
    
    stem_map = defaultdict(set)
    
    for word in tqdm(vocabulary, desc="Processing words", unit="words"):
        stem = stemmer.stem(word)
        
        doc = nlp(word)
        lemma = doc[0].lemma_.lower()
        
        stem_map[stem].add(lemma)
    
    final_map = dict(stem_map)
    
    print(f"Mapping complete! Created {len(final_map):,} unique stems")
    return final_map

def save_mapping(stem_map, filename="stem_to_lemma_map.pkl"):
    print(f"Saving mapping to {filename}...")
    
    with open(filename, 'wb') as f:
        pickle.dump(stem_map, f)
    
    total_stems = len(stem_map)
    total_lemmas = sum(len(lemmas) for lemmas in stem_map.values())
    avg_lemmas_per_stem = total_lemmas / total_stems if total_stems > 0 else 0
    
    print(f"✓ Stem-to-lemma mapping saved successfully!")
    print(f"  File: {filename}")
    print(f"  Total stems: {total_stems:,}")
    print(f"  Total lemmas: {total_lemmas:,}")
    print(f"  Average lemmas per stem: {avg_lemmas_per_stem:.2f}")
    
    return filename

def main():
    print("=" * 60)
    print("EU Parliament Stem-to-Lemma Mapping Creation")
    print("=" * 60)
    
    try:

        df = load_dataset()

        all_text = extract_all_text(df)

        vocabulary = extract_vocabulary(all_text)

        stemmer, nlp = initialize_nlp_tools()

        stem_map = build_stem_lemma_map(vocabulary, stemmer, nlp)

        output_file = save_mapping(stem_map)
        
        print(f"\nProcess completed successfully!")
        print(f"Output file: {output_file}")

        print(f"\nExample mappings (first 5):")
        for i, (stem, lemmas) in enumerate(list(stem_map.items())[:5]):
            lemma_list = sorted(list(lemmas))
            print(f"  '{stem}' → {lemma_list}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()