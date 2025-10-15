
# Phase 2 embedding extraction for EU Parliament discourse analysis.
# Main functions: extract_all_embeddings, setup_model, process_sentence_batch_vectorized, extract_keyword_embedding.
# Result: Converts keyword-relevant sentences into contextualized vector embeddings using RoBERTa/DistilRoBERTa,
# outputting structured embedding data for clustering and polarization analysis in Phase 3.


import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
import os
from collections import defaultdict
import re
from tqdm import tqdm
import nltk.stem

warnings.filterwarnings('ignore')

def load_stem_lemma_map():
    map_locations = [
        "stem_to_lemma_map.pkl",
        os.path.join("src", "stem_to_lemma_map.pkl"),
        os.path.join("data", "stem_to_lemma_map.pkl")
    ]
    
    for map_file in map_locations:
        if os.path.exists(map_file):
            
            with open(map_file, 'rb') as f:
                stem_map = pickle.load(f)
            
            return stem_map
    
    raise FileNotFoundError("stem_to_lemma_map.pkl not found. Run create_stem_lemma_map.py to generate it.")

def get_morphological_variations(keyword, stem_map, stemmer):

    keyword_stem = stemmer.stem(keyword.lower())

    variations = stem_map.get(keyword_stem, {keyword.lower()})

    variations = set(variations)
    variations.add(keyword.lower())
    
    return variations


MODEL_CONFIGS = {
    'roberta': {
        'name': 'roberta-base',
        'embedding_dim': 768,
        'batch_sizes': {'gpu_80gb': 8192*12, 'gpu_24gb': 4096*12, 'gpu_12gb': 2048*12, 'gpu_8gb': 1024*12, 'gpu_default': 512*12, 'cpu': 64},
        'vectorized_batch_sizes': {'gpu_80gb': 2048, 'gpu_24gb': 1024, 'gpu_12gb': 512, 'gpu_8gb': 256, 'gpu_default': 128, 'cpu': 64},
        'description': 'RoBERTa-base (768-dim, ~500MB, high accuracy)',
        'use_keyword_extraction': True,
        'use_vectorized': True
    },
    'distilroberta': {
        'name': 'distilroberta-base',
        'embedding_dim': 768,
        'batch_sizes': {'gpu_80gb': 16384*12, 'gpu_24gb': 8192*12, 'gpu_12gb': 4096*12, 'gpu_8gb': 2048*12, 'gpu_default': 1024*12, 'cpu': 128},
        'vectorized_batch_sizes': {'gpu_80gb': 4096, 'gpu_24gb': 2048, 'gpu_12gb': 1024, 'gpu_8gb': 512, 'gpu_default': 256, 'cpu': 128},
        'description': 'DistilRoBERTa-base (768-dim, ~300MB, fast & efficient)',
        'use_keyword_extraction': True,
        'use_vectorized': True
    }
}

def preallocate_gpu_memory(device, target_gb=120):

    if not torch.cuda.is_available():
        return None
        
    try:
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_properties.total_memory / (1024**3)
        
        if total_memory_gb >= 120:
            target_bytes = int(target_gb * 1024**3)
            preallocated_tensor = torch.empty(target_bytes // 4, dtype=torch.float32, device=device)
            return preallocated_tensor
        else:
            return None
            
    except Exception as e:
        return None

def setup_model(model_name='roberta'):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    actual_model_name = config['name']

    tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
    model = AutoModel.from_pretrained(actual_model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if hasattr(model, 'eval'):
        model.eval()
    

    batch_sizes = config['batch_sizes']
    vectorized_batch_sizes = config.get('vectorized_batch_sizes', batch_sizes)
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 100:
            batch_size = batch_sizes['gpu_80gb']
            vectorized_batch_size = vectorized_batch_sizes['gpu_80gb']
        elif gpu_memory_gb >= 24:
            batch_size = batch_sizes['gpu_24gb']
            vectorized_batch_size = vectorized_batch_sizes['gpu_24gb']
        elif gpu_memory_gb >= 12:
            batch_size = batch_sizes['gpu_12gb']
            vectorized_batch_size = vectorized_batch_sizes['gpu_12gb']
        elif gpu_memory_gb >= 8:
            batch_size = batch_sizes['gpu_8gb']
            vectorized_batch_size = vectorized_batch_sizes['gpu_8gb']
        else:
            batch_size = batch_sizes['gpu_default']
            vectorized_batch_size = vectorized_batch_sizes['gpu_default']

        preallocated_memory = None
        
    else:
        batch_size = batch_sizes['cpu']
        vectorized_batch_size = vectorized_batch_sizes['cpu']
        preallocated_memory = None
    
    return tokenizer, model, device, batch_size, vectorized_batch_size, config

def load_sentence_data(run_folder="../results", filename=None, theme_name=None, target_keyword="security", keyword_group=None):

    if filename is None:

        if keyword_group:
            keyword_part = "_".join(keyword_group)
        else:
            keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)
        
        if theme_name:
            filename = f'poc_{keyword_part}_{theme_name}_sentences.pkl'
        else:
            filename = f'poc_{keyword_part}_sentences.pkl'
    
    filepath = os.path.join(run_folder, "data", filename)
    
    with open(filepath, 'rb') as f:
        keyword_sentences = pickle.load(f)
    
    total_sentences = sum(len(sentences) for sentences in keyword_sentences.values())
    
    return keyword_sentences

def extract_keyword_embedding(sentence, keyword, tokenizer, model, device, keyword_group=None):
    if keyword_group:
        keywords_to_check = keyword_group
    else:
        keywords_to_check = [keyword] if isinstance(keyword, str) else keyword
    

    sentence_lower = sentence.lower()
    found_keywords = [kw for kw in keywords_to_check if kw.lower() in sentence_lower]
    if not found_keywords:
        return None
    

    active_keyword = found_keywords[0]
    

    try:
        inputs = tokenizer(
            sentence, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            truncation=True, 
            max_length=512, 
            padding=True
        )
        

        offset_mapping = inputs.pop('offset_mapping')[0]
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
    except Exception:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        offset_mapping = None

    with torch.no_grad():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        if gpu_memory_gb >= 78:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state.float()
        else:
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
    
    token_embeddings = []
    
    if offset_mapping is not None:
        #offset mapping for alignment
        sentence_lower = sentence.lower()
        keyword_lower = active_keyword.lower()

        #Find all keyword occurrences in the sentence
        keyword_positions = []
        start = 0
        while True:
            pos = sentence_lower.find(keyword_lower, start)
            if pos == -1:
                break
            #Check if it's a whole word
            if (pos == 0 or not sentence_lower[pos-1].isalnum()) and \
               (pos + len(keyword_lower) == len(sentence_lower) or not sentence_lower[pos + len(keyword_lower)].isalnum()):
                keyword_positions.append((pos, pos + len(keyword_lower)))
            start = pos + 1
        
        # Map character positions to tokens using offset mapping
        for char_start, char_end in keyword_positions:
            token_indices = []
            
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue

                if not (token_end <= char_start or token_start >= char_end):
                    token_indices.append(token_idx)
            
            if token_indices:
                #Extract embeddings for the keyword tokens
                keyword_token_embeddings = hidden_states[0, token_indices, :]
                avg_embedding = torch.mean(keyword_token_embeddings, dim=0)
                token_embeddings.append(avg_embedding)
    
    else:
        keyword_tokens = tokenizer.encode(active_keyword, add_special_tokens=False)
        input_ids = inputs['input_ids'][0]

        for i in range(len(input_ids) - len(keyword_tokens) + 1):
            if input_ids[i:i+len(keyword_tokens)].tolist() == keyword_tokens:
                keyword_embeddings = hidden_states[0, i:i+len(keyword_tokens), :]
                avg_embedding = torch.mean(keyword_embeddings, dim=0)
                token_embeddings.append(avg_embedding)
    
    if token_embeddings:
        final_embedding = torch.mean(torch.stack(token_embeddings), dim=0)
        return final_embedding.cpu().numpy()
    
    return None

def vectorized_keyword_search(input_ids_batch, keyword_tokens, device):

    batch_size, seq_len = input_ids_batch.shape
    keyword_len = len(keyword_tokens)
    
    if keyword_len == 0 or keyword_len > seq_len:
        return []

    keyword_tensor = torch.tensor(keyword_tokens, device=device).unsqueeze(0)

    windows = input_ids_batch.unfold(1, keyword_len, 1)
    
    if windows.size(1) == 0:
        return []
    keyword_expanded = keyword_tensor.unsqueeze(1).expand(batch_size, windows.size(1), keyword_len)

    matches = (windows == keyword_expanded).all(dim=2)

    batch_indices, position_indices = torch.where(matches)

    keyword_positions = []
    for i in range(len(batch_indices)):
        batch_idx = batch_indices[i].item()
        start_pos = position_indices[i].item()
        keyword_positions.append((batch_idx, start_pos))
    
    return keyword_positions

def gpu_aggregate_embeddings(hidden_states, keyword_positions, keyword_len, device):

    if not keyword_positions:
        return {}
    
    batch_embeddings = {}
    
    for batch_idx, start_pos in keyword_positions:
        end_pos = start_pos + keyword_len

        keyword_embeddings = hidden_states[batch_idx, start_pos:end_pos, :]

        avg_embedding = torch.mean(keyword_embeddings, dim=0)

        if batch_idx not in batch_embeddings:
            batch_embeddings[batch_idx] = []
        batch_embeddings[batch_idx].append(avg_embedding)

    final_embeddings = {}
    for batch_idx, embeddings_list in batch_embeddings.items():
        if len(embeddings_list) == 1:
            final_embeddings[batch_idx] = embeddings_list[0]
        else:
            stacked = torch.stack(embeddings_list, dim=0)
            final_embeddings[batch_idx] = torch.mean(stacked, dim=0)
    
    return final_embeddings

def extract_keyword_embeddings_batch_vectorized(sentences, keyword, tokenizer, model, device):
    valid_sentences = []
    valid_indices = []
    
    keyword_lower = keyword.lower()
    for i, sentence in enumerate(sentences):
        if keyword_lower in sentence.lower():
            valid_sentences.append(sentence)
            valid_indices.append(i)
    
    if not valid_sentences:
        return []

    batch_inputs = tokenizer(
        valid_sentences,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    input_ids_batch = batch_inputs['input_ids']

    keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
    
    if not keyword_tokens:
        return []

    try:
        with torch.no_grad():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            if gpu_memory_gb >= 78:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch_inputs)
                    hidden_states = outputs.last_hidden_state.float()
            else:
                outputs = model(**batch_inputs)
                hidden_states = outputs.last_hidden_state
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU out of memory in vectorized processing with batch size {len(valid_sentences)}. Try reducing batch size.")
        else:
            raise e
    

    try:
        keyword_positions = vectorized_keyword_search(input_ids_batch, keyword_tokens, device)
    except Exception as e:
        import logging
        logging.warning(f"Vectorized keyword search failed: {str(e)[:100]}")
        keyword_positions = []

    try:
        batch_embeddings_gpu = gpu_aggregate_embeddings(hidden_states, keyword_positions, len(keyword_tokens), device)
    except Exception as e:
        import logging
        logging.warning(f"GPU aggregation failed: {str(e)[:100]}")
        batch_embeddings_gpu = {}

    final_embeddings = []
    for i in range(len(valid_sentences)):
        if i in batch_embeddings_gpu:
            embedding = batch_embeddings_gpu[i].cpu().numpy()
            final_embeddings.append(embedding)
        else:
            try:
                fallback_embedding = extract_keyword_embedding(valid_sentences[i], keyword, tokenizer, model, device)
                if fallback_embedding is not None:
                    final_embeddings.append(fallback_embedding)
            except Exception as e:
                import logging
                logging.warning(f"Failed to extract embedding for sentence: {str(e)[:100]}")
                continue
    
    return final_embeddings

def extract_keyword_embeddings_batch(sentences, keyword, tokenizer, model, device):

    valid_sentences = []
    valid_indices = []
    
    for i, sentence in enumerate(sentences):
        if keyword.lower() in sentence.lower():
            valid_sentences.append(sentence)
            valid_indices.append(i)
    
    if not valid_sentences:
        return []
    

    try:
        batch_inputs = tokenizer(
            valid_sentences,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding=True
        )

        offset_mappings = batch_inputs.pop('offset_mapping')  # Keep on CPU
        batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
        use_offset_mapping = True
        
    except Exception:
        batch_inputs = tokenizer(
            valid_sentences,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
        offset_mappings = None
        use_offset_mapping = False

    with torch.no_grad():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        if gpu_memory_gb >= 78:
            with torch.cuda.amp.autocast():
                outputs = model(**batch_inputs)
                hidden_states = outputs.last_hidden_state.float()
        else:
            outputs = model(**batch_inputs)
            hidden_states = outputs.last_hidden_state
    

    batch_embeddings = []
    keyword_lower = keyword.lower()
    
    for batch_idx, sentence in enumerate(valid_sentences):
        sentence_embeddings = []
        
        if use_offset_mapping:
            sentence_lower = sentence.lower()
            offset_mapping = offset_mappings[batch_idx]

            keyword_positions = []
            start = 0
            while True:
                pos = sentence_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                if (pos == 0 or not sentence_lower[pos-1].isalnum()) and \
                   (pos + len(keyword_lower) == len(sentence_lower) or not sentence_lower[pos + len(keyword_lower)].isalnum()):
                    keyword_positions.append((pos, pos + len(keyword_lower)))
                start = pos + 1

            for char_start, char_end in keyword_positions:
                token_indices = []
                
                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == 0 and token_end == 0:
                        continue

                    if not (token_end <= char_start or token_start >= char_end):
                        token_indices.append(token_idx)
                
                if token_indices:

                    keyword_token_embeddings = hidden_states[batch_idx, token_indices, :]
                    avg_embedding = torch.mean(keyword_token_embeddings, dim=0)
                    sentence_embeddings.append(avg_embedding)
        
        else:

            keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
            input_ids = batch_inputs['input_ids'][batch_idx]

            for i in range(len(input_ids) - len(keyword_tokens) + 1):
                if input_ids[i:i+len(keyword_tokens)].tolist() == keyword_tokens:
                    keyword_embeddings = hidden_states[batch_idx, i:i+len(keyword_tokens), :]
                    avg_embedding = torch.mean(keyword_embeddings, dim=0)
                    sentence_embeddings.append(avg_embedding)
        
        if sentence_embeddings:
            final_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
            batch_embeddings.append(final_embedding.cpu().numpy())
    
    return batch_embeddings


def process_sentence_batch_with_accumulation(sentences, keyword, tokenizer, model, device, batch_size=256,
                                             accumulation_batches=10):

    embeddings = []

    batch_iterator = range(0, len(sentences), batch_size)
    total_batches = len(list(batch_iterator))

    if total_batches > 1:
        iterator = tqdm(range(0, len(sentences), batch_size), desc=f"Processing with accumulation", leave=False)
    else:
        iterator = range(0, len(sentences), batch_size)

    for i in iterator:
        batch_sentences = sentences[i:i + batch_size]

        try:

            batch_embeddings = extract_keyword_embeddings_batch_vectorized_with_gpu_retention(
                batch_sentences, keyword, tokenizer, model, device)

            embeddings.extend(batch_embeddings)

        except Exception as e:
            try:
                batch_embeddings = extract_keyword_embeddings_batch(batch_sentences, keyword, tokenizer, model, device)
                embeddings.extend(batch_embeddings)
            except Exception as e2:
                continue

    return embeddings


def extract_keyword_embeddings_batch_vectorized_with_gpu_retention(sentences, keyword, tokenizer, model, device):

    result = extract_keyword_embeddings_batch_vectorized(sentences, keyword, tokenizer, model, device)
    return result

def process_sentence_batch_vectorized_multistream(sentences, keyword, tokenizer, model, device, batch_size=256, num_streams=4):

    import torch.cuda
    embeddings = []

    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    batch_iterator = range(0, len(sentences), batch_size)
    total_batches = len(list(batch_iterator))

    batch_futures = []
    stream_idx = 0
    
    if total_batches > 1:
        iterator = tqdm(range(0, len(sentences), batch_size), desc=f"Processing multi-stream batches", leave=False)
    else:
        iterator = range(0, len(sentences), batch_size)
    
    for i in iterator:
        batch_sentences = sentences[i:i+batch_size]
        current_stream = streams[stream_idx % num_streams]

        with torch.cuda.stream(current_stream):
            try:
                batch_embeddings = extract_keyword_embeddings_batch_vectorized(batch_sentences, keyword, tokenizer, model, device)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                try:
                    batch_embeddings = extract_keyword_embeddings_batch(batch_sentences, keyword, tokenizer, model, device)
                    embeddings.extend(batch_embeddings)
                except Exception as e2:
                    for sentence in batch_sentences:
                        try:
                            embedding = extract_keyword_embedding(sentence, keyword, tokenizer, model, device)
                            if embedding is not None:
                                embeddings.append(embedding)
                        except Exception as e3:
                            continue
        
        stream_idx += 1

        if stream_idx % (num_streams * 2) == 0:
            for stream in streams:
                stream.synchronize()

    for stream in streams:
        stream.synchronize()
    
    return embeddings

def process_sentence_batch_vectorized(sentences, keyword, tokenizer, model, device, batch_size=256):

    embeddings = []
    
    batch_iterator = range(0, len(sentences), batch_size)
    total_batches = len(list(batch_iterator))
    
    if total_batches > 1:
        iterator = tqdm(range(0, len(sentences), batch_size), desc=f"Processing vectorized batches", leave=False)
    else:
        iterator = range(0, len(sentences), batch_size)
    
    for i in iterator:
        batch_sentences = sentences[i:i+batch_size]
        
        try:
            batch_embeddings = extract_keyword_embeddings_batch_vectorized(batch_sentences, keyword, tokenizer, model, device)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            try:
                batch_embeddings = extract_keyword_embeddings_batch(batch_sentences, keyword, tokenizer, model, device)
                embeddings.extend(batch_embeddings)
            except Exception as e2:
                batch_embeddings = []
                for sentence in batch_sentences:
                    try:
                        embedding = extract_keyword_embedding(sentence, keyword, tokenizer, model, device)
                        if embedding is not None:
                            batch_embeddings.append(embedding)
                    except Exception as e3:
                        continue
                embeddings.extend(batch_embeddings)
    
    return embeddings

def process_sentence_batch(sentences, keyword, tokenizer, model, device, batch_size=16):

    embeddings = []
    
    batch_iterator = range(0, len(sentences), batch_size)
    total_batches = len(list(batch_iterator))
    
    if total_batches > 1:
        iterator = tqdm(range(0, len(sentences), batch_size), desc=f"Processing batches", leave=False)
    else:
        iterator = range(0, len(sentences), batch_size)
    
    for i in iterator:
        batch_sentences = sentences[i:i+batch_size]
        
        try:
            batch_embeddings = extract_keyword_embeddings_batch(batch_sentences, keyword, tokenizer, model, device)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            batch_embeddings = []
            for sentence in batch_sentences:
                embedding = extract_keyword_embedding(sentence, keyword, tokenizer, model, device)
                if embedding is not None:
                    batch_embeddings.append(embedding)
            embeddings.extend(batch_embeddings)
    
    return embeddings


def extract_all_embeddings(keyword_sentences, tokenizer, model, device, run_folder="../results", keyword="security", batch_size=16, vectorized_batch_size=1024, theme_name=None, config=None):

    if theme_name:
        pass
    else:
        pass
    
    use_keyword_extraction = config['use_keyword_extraction'] if config else True
    use_vectorized = config.get('use_vectorized', False) if config else False
    
    if use_vectorized and use_keyword_extraction:
        pass
    elif use_keyword_extraction:
        pass
    else:
        pass
    
    try:
        stem_map = load_stem_lemma_map()
        stemmer = nltk.stem.SnowballStemmer('english')
    except Exception as e:
        stem_map = None
        stemmer = None
    
    if stem_map and stemmer:
        keyword_variations = get_morphological_variations(keyword, stem_map, stemmer)
    else:
        keyword_variations = {keyword.lower()}
    
    tagged_sentences = []
    total_input_sentences = 0
    
    for (term, group), sentences in keyword_sentences.items():
        total_input_sentences += len(sentences)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for variation in keyword_variations:
                if variation in sentence_lower:
                    tagged_sentences.append(((term, group), sentence, variation))
                    break
    
    if len(tagged_sentences) == 0:
        return {}
    
    sentences_by_keyword = defaultdict(list)
    
    for original_group_key, sentence, found_keyword in tagged_sentences:
        sentences_by_keyword[found_keyword].append((sentence, original_group_key))
    
    for variation, items in sentences_by_keyword.items():
        pass

    all_processed_embeddings = []
    total_processed = 0
    
    keyword_blocks = list(sentences_by_keyword.items())
    with tqdm(total=len(keyword_blocks), desc="Processing keyword blocks", mininterval=1.0) as pbar:
        for keyword_variation, sentence_items in keyword_blocks:
            sentences = [sentence for sentence, _ in sentence_items]
            
            
            if use_vectorized:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
                if gpu_memory_gb < 200:
                    if len(sentences) > vectorized_batch_size * 4:
                        embeddings = process_sentence_batch_with_accumulation(sentences, keyword_variation, tokenizer, model, device, vectorized_batch_size, accumulation_batches=20)
                    elif len(sentences) > vectorized_batch_size:
                        embeddings = process_sentence_batch_vectorized_multistream(sentences, keyword_variation, tokenizer, model, device, vectorized_batch_size, num_streams=4)
                    else:
                        embeddings = process_sentence_batch_vectorized(sentences, keyword_variation, tokenizer, model, device, vectorized_batch_size)
                else:
                    embeddings = process_sentence_batch_vectorized(sentences, keyword_variation, tokenizer, model, device, vectorized_batch_size)
            else:
                embeddings = process_sentence_batch(sentences, keyword_variation, tokenizer, model, device, batch_size)
            
            for i, embedding in enumerate(embeddings):
                if i < len(sentence_items):
                    _, original_group_key = sentence_items[i]
                    all_processed_embeddings.append((embedding, original_group_key))
                    total_processed += 1
            
            pbar.update(1)
    
    final_grouped_embeddings = defaultdict(list)
    
    for embedding, original_group_key in all_processed_embeddings:
        final_grouped_embeddings[original_group_key].append(embedding)
    

    final_embeddings = dict(final_grouped_embeddings)

    return final_embeddings

def save_embeddings(embeddings, run_folder="../results", filename=None, theme_name=None, target_keyword="security", model_name="roberta", keyword_group=None):

    if filename is None:

        if keyword_group:
            keyword_part = "_".join(keyword_group)
        else:
            keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)
        
        if theme_name:
            filename = f'poc_{keyword_part}_{theme_name}_{model_name}_raw_embeddings.pkl'
        else:
            filename = f'poc_{keyword_part}_{model_name}_raw_embeddings.pkl'
    
    filepath = os.path.join(run_folder, "data", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

    return filepath

def main(run_folder=None, theme_name=None, temporal_unit='term', target_keyword='security', model_name='roberta', keyword_group=None):

    if run_folder is None:
        run_folder = "../results"
    
    if theme_name:
        pass
    else:
        pass
    temporal_desc = "individual years" if temporal_unit == 'year' else "5-year parliamentary terms"
    
    try:
        #Setup model with vectorized optimization
        tokenizer, model, device, recommended_batch_size, vectorized_batch_size, config = setup_model(model_name)
        
        # Load sentence data from Phase 1
        keyword_sentences = load_sentence_data(run_folder, theme_name=theme_name, target_keyword=target_keyword, keyword_group=keyword_group)
        
        #Extract and aggregate embeddings with vectorized processing
        final_embeddings = extract_all_embeddings(
            keyword_sentences, tokenizer, model, device, run_folder, 
            keyword=target_keyword, 
            batch_size=recommended_batch_size,
            vectorized_batch_size=vectorized_batch_size,
            theme_name=theme_name, 
            config=config
        )
        
        #Save final embeddings
        output_file = save_embeddings(final_embeddings, run_folder, theme_name=theme_name, target_keyword=target_keyword, model_name=model_name, keyword_group=keyword_group)
        
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()