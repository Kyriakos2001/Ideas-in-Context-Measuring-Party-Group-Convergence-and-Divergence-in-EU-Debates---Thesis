# Phase 1 data preparation for EU Parliament discourse analysis.
# Main functions: create_keyword_slices, segment_sentences, lemmatize_sentences, get_morphological_variations.
# Result: Extracts keyword-relevant sentences from raw EU Parliament speeches, applies thematic filters,
# and outputs structured data for embedding analysis in Phase 2.

import pandas as pd
import nltk
import pickle
import re
import os
import subprocess
import sys
import multiprocessing as mp
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


tqdm.pandas(desc="Processing", mininterval=1.0, miniters=None)

# Cache version for sentence DataFrame with lemmas
CACHE_VERSION = 2

# Thematic Filters Configuration
THEMATIC_FILTERS = {
    'security': ['strategy', 'cyber', 'infrastructure', 'disinformation', 'intelligence', 'terrorism', 'counterterrorism', 'protection', 'resilience', 'surveillance', 'controls', 'crime', 'deterrence', 'screening', 'interoperability'],
    'health': ['COVID', 'WHO', 'pandemic', 'epidemic', 'healthcare', 'vaccination', 'pharmaceutical', 'medical', 'devices', 'mental', 'disease', 'care', 'elderly', 'transplant', 'crisis', 'heart', 'cancer', 'HIV', 'STD', 'AIDS', 'CHD', 'ACHD', 'ADHD', 'OCD'],
    'war': ['invasion', 'ceasefire', 'sanctions', 'arms', 'embargo', 'crimes', 'humanitarian', 'law', 'territory', 'annexation', 'mobilization', 'drones', 'refugees', 'escalation', 'construction', 'reconstruction', 'military', 'defence', 'army', 'Ukraine', 'Russia', 'Israel', 'Palestine', 'Gaza', 'POW', 'prisoners', 'peacekeepers', 'troops', 'monitoring', 'occupation', 'peace'],
    'enlargement': ['accession', 'candidate', 'country', 'status', 'criteria', 'chapters', 'negotiations', 'frameworks', 'laws', 'reports', 'progress', 'veto', 'unanimity', 'Montenegro', 'Ukraine', 'Moldova', 'Iceland', 'euro', 'referendum', 'immigration', 'minorities', 'reforms', 'judiciary', 'border'],
    'defence': ['military', 'army', 'cyber', 'security', 'training', 'missions', 'R&D', 'NATO'],
    'environment': ['water', 'nature', 'biodiversity', 'pollution', 'waste', 'plastic', 'soil', 'agriculture', 'farmers', 'conservation', 'restoration', 'reforestation', 'wetlands', 'air', 'quality', 'remediation', 'ecology'],
    'immigration': ['asylum', 'refugees', 'immigrants', 'rights', 'humanitarian', 'status', 'work', 'welfare', 'benefits', 'protests', 'strikes', 'racism', 'Islam', 'Muslims', 'Christians', 'Christianity', 'hotspots', 'camps', 'concentration', 'terrorism', 'hotels', 'family', 'reunification', 'crime', 'temporary', 'protection', 'Frontex', 'Schengen', 'smuggling', 'visa', 'skills', 'labor', 'resettlement', 'religion', 'discrimination', 'culture', 'language', 'education', 'Blue-Card', 'pathways', 'citizenship', 'UN', 'UNHCR'],
    'integration': ['language', 'civic', 'culture', 'skills', 'communities', 'inclusion', 'education', 'recognition', 'mentoring', 'housing', 'employment', 'discrimination', 'cohesion', 'outreach', 'credential'],
    'economy': ['fiscal', 'financial', 'single', 'market', 'SME', 'policy', 'competition', 'capital', 'state', 'aid', 'industry', 'tariffs', 'resources', 'employment', 'unemployment', 'regulations', 'laws', 'production', 'inflation', 'reforms', 'ECB', 'social', 'welfare', 'benefits', 'poverty', 'Blue-Card', 'pathways', 'citizenship', 'innovation', 'wage', 'salary', 'income'],
    'energy': ['environment', 'nuclear', 'electrical', 'renewable', 'coal', 'gas', 'fuel', 'petrol', 'windmills', 'LNG', 'tax', 'carbon', 'cap', 'price', 'solar', 'hydrogen', 'grid', 'pumps'],
    'digital': ['AI', 'DSA', 'DMA', 'GDPR', 'platforms', 'data', 'chips', 'algorithms', 'centers', 'cybersecurity', 'government', 'e-governance', 'transparency', 'models', 'moderation', 'content', 'sovereignty', 'accountability', 'fines'],
    'trade': ['tariffs', 'dumping', 'safeguards', 'export', 'import', 'prices', 'investment', 'supply', 'chains', 'agreements', 'quotas', 'market', 'access', 'services', 'competitiveness', 'retaliation', 'sanctions'],
    'transportation': ['infrastructure', 'road', 'tolls', 'transport', 'cars', 'bus', 'metro', 'tram', 'train', 'railway', 'corridors', 'bike', 'logistics', 'delays', 'sidewalks', 'pedestrians', 'drivers', 'vignette', 'aviation', 'airports', 'maritime', 'ferries', 'safety', 'license']
}


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def ensure_stem_lemma_map():
    map_file = "stem_to_lemma_map.pkl"
    
    if os.path.exists(map_file):
        return map_file
    
    print("Creating stem-to-lemma mapping (this may take several minutes)...")

    result = subprocess.run([sys.executable, "create_stem_lemma_map.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        return map_file
    else:
        raise RuntimeError(f"Stem-to-lemma mapping creation failed: {result.stderr}")

def load_stem_lemma_map():
    map_file = ensure_stem_lemma_map()
    
    with open(map_file, 'rb') as f:
        stem_map = pickle.load(f)
    
    return stem_map

def get_morphological_variations(keyword, stem_map, stemmer):

    keyword_stem = stemmer.stem(keyword.lower())

    variations = stem_map.get(keyword_stem, {keyword.lower()})

    variations = set(variations)
    variations.add(keyword.lower())
    
    return variations

def extract_valid_political_groups(df):
    all_groups = df['epg_short'].dropna().unique().tolist()
    all_groups.sort()
    
    
    return all_groups

def setup_poc_parameters(df, target_keyword="security"):
    TARGET_GROUPS = extract_valid_political_groups(df)
    return target_keyword, TARGET_GROUPS

def load_raw_data():

    import pyreadr
    result = pyreadr.read_r("data/EUPDCorp_1999-2024_v1.RDS")
    df = result[None]
    df['word_count'] = df['speech_en'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    return df

def filter_data_by_groups(df, target_groups):
    filtered_df = df[df['epg_short'].isin(target_groups)].copy()
    filtered_df = filtered_df.dropna(subset=['speech_en', 'term_no'])
    return filtered_df

def get_segmented_sentences(filtered_df, run_folder):
    cache_locations = [
        "sentence_df_cache.pkl",  # Root level (precomputed)
        os.path.join(run_folder, "data", "sentence_df_cache.pkl")  # Run-specific
    ]
    
    for cache_file in cache_locations:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            if isinstance(cached_data, dict) and cached_data.get('version') == CACHE_VERSION:
                sentence_df = cached_data['data']
                return sentence_df
            else:
                pass


    sentence_df = segment_sentences(filtered_df)

    sentence_df = lemmatize_sentences(sentence_df)

    cache_file = os.path.join(run_folder, "data", "sentence_df_cache.pkl")

    data_dir = os.path.dirname(cache_file)
    os.makedirs(data_dir, exist_ok=True)
    

    cache_data = {
        'version': CACHE_VERSION,
        'data': sentence_df
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return sentence_df

def process_speech_chunk_segmentation(chunk):
    import nltk
    # Initialize NLTK tokenizer in each process
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    sent_tokenizer = nltk.sent_tokenize
    
    results = []
    for idx, row in chunk.iterrows():
        speech_text = row['speech_en'].strip()
        sentences = sent_tokenizer(speech_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:  # Filter out very short sentences
                results.append({
                    'sentence': sentence,
                    'term_no': row['term_no'],
                    'year': row['year'],
                    'epg_short': row['epg_short'],
                    'original_speech_idx': idx
                })
    return results

def segment_sentences(df):
    print("\nPerforming sentence segmentation with multiprocessing, if on google Collab, run this on TPU for max cores...")

    sent_tokenizer = nltk.sent_tokenize
    test_sentences = sent_tokenizer("This is a test. This is another test.")
    print(f"NLTK tokenizer initialized. Test produced {len(test_sentences)} sentences.")

    valid_mask = df['speech_en'].notna() & (df['speech_en'].str.len() > 10)
    valid_speeches = df[valid_mask]

    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for sentence segmentation")

    chunk_size = max(1, len(valid_speeches) // (num_cores * 4))
    chunks = [valid_speeches.iloc[i:i + chunk_size] for i in range(0, len(valid_speeches), chunk_size)]
    
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} speeches each")

    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_speech_chunk_segmentation, chunks),
            total=len(chunks),
            desc="Segmenting speeches"
        ))

    all_sentences = []
    for chunk_result in results:
        all_sentences.extend(chunk_result)

    sentence_df = pd.DataFrame(all_sentences)
    
    return sentence_df

def lemmatize_sentences(sentence_df):
    import spacy

    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError:
        raise

    num_cores = mp.cpu_count()
    
    sentences = sentence_df['sentence'].tolist()

    sentence_lemmas = []

    docs = nlp.pipe(
        [sentence.lower() for sentence in sentences], 
        n_process=num_cores,
        batch_size=1000
    )
    
    with tqdm(total=len(sentences), desc="Lemmatizing sentences", mininterval=1.0) as pbar:
        for doc in docs:
            lemmas = {token.lemma_ for token in doc 
                     if token.is_alpha and len(token.lemma_) > 1}
            sentence_lemmas.append(lemmas)
            pbar.update(1)

    sentence_df['sentence_lemmas'] = sentence_lemmas
    
    return sentence_df

def create_keyword_slices(sentence_df, target_keyword, run_folder="../results", filter_name=None, filter_keywords=None, temporal_unit='term', keyword_group=None, use_morphological_variations=True, min_sentence_threshold=0):

    # Load stem-to-lemma mapping and initialize stemmer
    stem_map = None
    stemmer = None
    if use_morphological_variations:
        stem_map = load_stem_lemma_map()
        import nltk.stem
        stemmer = nltk.stem.SnowballStemmer('english')

    if keyword_group:
        keywords_to_search = keyword_group
        keyword_display = " OR ".join(keyword_group)
        is_group_analysis = True
    else:
        keywords_to_search = [target_keyword] if isinstance(target_keyword, str) else target_keyword
        keyword_display = " OR ".join(keywords_to_search) if isinstance(keywords_to_search, list) else str(target_keyword)
        is_group_analysis = len(keywords_to_search) > 1 if isinstance(keywords_to_search, list) else False

    keyword_lemmas_set = set()
    if use_morphological_variations and stem_map and stemmer:

        for keyword in keywords_to_search:
            variations = get_morphological_variations(keyword, stem_map, stemmer)
            keyword_lemmas_set.update(variations)
            print(f"'{keyword}' expanded to: {', '.join(sorted(variations))}")
        
        print(f"Total keyword lemmas for filtering: {len(keyword_lemmas_set)}")
    else:

        keyword_lemmas_set = {kw.lower() for kw in keywords_to_search}
        print(f"Using exact keyword lemmas: {', '.join(sorted(keyword_lemmas_set))}")
    
    # Pre-lemmatize filter keywords for thematic filtering
    filter_lemmas_set = set()
    if filter_keywords:
        if use_morphological_variations and stem_map and stemmer:
            for filter_kw in filter_keywords:
                variations = get_morphological_variations(filter_kw, stem_map, stemmer)
                filter_lemmas_set.update(variations)
        else:
            filter_lemmas_set = {kw.lower() for kw in filter_keywords}
    
    if filter_name:
        print(f"\nFiltering sentences containing '{keyword_display}' in {filter_name} context...")
        context_info = f" ({filter_name})"
    else:
        print(f"\nFiltering sentences containing '{keyword_display}'...")
        context_info = ""

    if filter_name:
        print(f"Step 1: Filtering sentences containing '{keyword_display}' (using lemma sets)...")
    else:
        print(f"Filtering sentences containing '{keyword_display}' (using lemma sets)...")
    
    contains_keyword = sentence_df['sentence_lemmas'].apply(
        lambda lemmas: not lemmas.isdisjoint(keyword_lemmas_set)
    )
    
    keyword_sentences = sentence_df[contains_keyword]
    
    if is_group_analysis:
        print(f"Found {len(keyword_sentences):,} sentences containing any of: {keyword_display}")
    else:
        print(f"Found {len(keyword_sentences):,} sentences containing '{keyword_display}'")
    

    if filter_keywords:
        print(f"Step 2: Applying {filter_name} thematic filter (using lemma sets)...")

        thematic_filter = keyword_sentences['sentence_lemmas'].apply(
            lambda lemmas: not lemmas.isdisjoint(filter_lemmas_set)
        )
        keyword_sentences = keyword_sentences[thematic_filter]
        
        print(f"After thematic filtering: {len(keyword_sentences):,} sentences")
        print(f"Thematic lemmas: {len(filter_lemmas_set)} unique forms")
        
        final_sentence_count = len(keyword_sentences)
    else:
        final_sentence_count = len(keyword_sentences)
    
    print(f"Final sentences for analysis{context_info}: {final_sentence_count:,}")
    
    # {(temporal_unit, group): [sentences]}
    grouped_sentences = defaultdict(list)
    
    temporal_col = 'year' if temporal_unit == 'year' else 'term_no'
    temporal_desc = 'year' if temporal_unit == 'year' else 'term'
    print(f"Creating sentence groups by ({temporal_desc}, political_group)...")
    
    for idx, row in tqdm(keyword_sentences.iterrows(), total=len(keyword_sentences), desc="Grouping sentences"):
        temporal_value = int(row[temporal_col])
        key = (temporal_value, row['epg_short'])
        grouped_sentences[key].append(row['sentence'])

    grouped_sentences = dict(grouped_sentences)
    
    print(f"\nCreated sentence slices for {len(grouped_sentences)} ({temporal_desc}, group) combinations:")
    for key, sentences in grouped_sentences.items():
        temporal_val, group = key
        print(f"  {temporal_desc.title()} {temporal_val}, {group}: {len(sentences)} sentences")

    total_sentences = sum(len(sentences) for sentences in grouped_sentences.values())
    print(f"\nFinal total sentences: {total_sentences:,}")
    

    if min_sentence_threshold > 0:
        print(f"\nApplying minimum sentence threshold of {min_sentence_threshold}...")

        filtered_grouped_sentences = {}
        excluded_groups = []

        for key, sentences in grouped_sentences.items():
            if len(sentences) >= min_sentence_threshold:
                filtered_grouped_sentences[key] = sentences
            else:
                temporal_val, group = key
                excluded_groups.append(f"Term/Year {temporal_val}, {group} ({len(sentences)} sentences)")

        if excluded_groups:
            print(f"Excluded {len(excluded_groups)} groups that did not meet the threshold:")
            for excluded in excluded_groups[:10]:
                print(f"  - {excluded}")
            if len(excluded_groups) > 10:
                print(f"  ... and {len(excluded_groups) - 10} more")

        print(f"Kept {len(filtered_grouped_sentences)} groups that met the threshold.")
        final_sentences_dict = dict(filtered_grouped_sentences)

    else:
        print("\nNo minimum sentence threshold applied (threshold is 0).")
        final_sentences_dict = dict(grouped_sentences)
    
    return final_sentences_dict

def save_intermediate_result(keyword_sentences, run_folder="../results", filename=None, filter_name=None, target_keyword="security", keyword_group=None):

    #Save Intermediate Result: Save dictionary to pickle file


    if filename is None:
        if keyword_group:
            keyword_part = "_".join(keyword_group)
        else:
            keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)
        
        if filter_name:
            filename = f'poc_{keyword_part}_{filter_name}_sentences.pkl'
        else:
            filename = f'poc_{keyword_part}_sentences.pkl'

    filepath = os.path.join(run_folder, "data", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(keyword_sentences, f)
    
    
    
    return filepath

def main(run_folder=None, theme_name=None, theme_keywords=None, temporal_unit='term', target_keyword='security', keyword_group=None, use_morphological_variations=True, min_sentence_threshold=0):

    if run_folder is None:
        run_folder = "../results"


    if theme_name:
        pass
    else:
        pass
    
    if keyword_group:
        pass
    else:
        pass
    
    #Load raw data first to analyze political groups
    raw_df = load_raw_data()
    
    #Setup parameters with dynamic group extraction
    target_keyword, target_groups = setup_poc_parameters(raw_df, target_keyword)
    
    #Filter data by extracted groups
    filtered_df = filter_data_by_groups(raw_df, target_groups)
    
    # Sentence segmentation (with caching)
    sentence_df = get_segmented_sentences(filtered_df, run_folder)
    
    # Create keyword slices
    keyword_sentences = create_keyword_slices(sentence_df, target_keyword, run_folder, theme_name, theme_keywords, temporal_unit, keyword_group, use_morphological_variations, min_sentence_threshold)
    
    #Save intermediate result
    output_file = save_intermediate_result(keyword_sentences, run_folder, filter_name=theme_name, target_keyword=target_keyword, keyword_group=keyword_group)
    

if __name__ == "__main__":
    main()