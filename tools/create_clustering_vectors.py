"""
utility that generates concept vectors for each theme defined in
src/clustering_themes.py using the specified transformer model. The resulting
vectors are used for semantic similarity-based cluster labeling in last phase.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering_themes import CLUSTERING_THEMES
from poc_phase2_embeddings import setup_model

def extract_keyword_embedding(text, keyword, tokenizer, model, device):

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

    keyword_token_indices = []
    keyword_lower = keyword.lower()

    for i, token in enumerate(tokens):
        clean_token = token.replace('Ġ', '').lower()

        if (clean_token == keyword_lower or
            keyword_lower.startswith(clean_token) or
            clean_token in keyword_lower):
            keyword_token_indices.append(i)

    if not keyword_token_indices:
        for i, token in enumerate(tokens):
            clean_token = token.replace('Ġ', '').lower()
            if len(clean_token) > 2 and clean_token in keyword_lower:
                keyword_token_indices.append(i)

    if not keyword_token_indices:
        non_special_indices = []
        for i, token in enumerate(tokens):
            if token not in ['<s>', '</s>', '[CLS]', '[SEP]', '<pad>']:
                non_special_indices.append(i)
        keyword_token_indices = non_special_indices

    if keyword_token_indices:
        keyword_embeddings = embeddings[keyword_token_indices]
        keyword_embedding = torch.mean(keyword_embeddings, dim=0)
    else:
        keyword_embedding = torch.mean(embeddings[1:-1], dim=0)

    return keyword_embedding.cpu().numpy()

def generate_theme_concept_vector(theme_name, keywords, tokenizer, model, device):

    keyword_embeddings = []

    for keyword in tqdm(keywords, desc=f"Processing {theme_name}", leave=False):
        context_template = f"A discussion about {keyword}."

        try:
            embedding = extract_keyword_embedding(context_template, keyword, tokenizer, model, device)
            keyword_embeddings.append(embedding)
        except Exception as e:
            continue

    if not keyword_embeddings:
        return None

    concept_vector = np.mean(keyword_embeddings, axis=0)

    return concept_vector

def main():
    parser = argparse.ArgumentParser(description='Generate concept vectors for clustering themes')
    parser.add_argument('--model',
                       choices=['roberta', 'distilroberta'],
                       required=True,
                       help='Transformer model to use for embedding generation')

    args = parser.parse_args()

    tokenizer, model, device, _, _, _ = setup_model(args.model)

    concept_vectors = {}

    for theme_name, keywords in CLUSTERING_THEMES.items():
        concept_vector = generate_theme_concept_vector(theme_name, keywords, tokenizer, model, device)

        if concept_vector is not None:
            concept_vectors[theme_name] = concept_vector
        else:
            pass
    output_filename = f"clustering_vectors_{args.model}.pkl"
    output_path = os.path.join("src", output_filename)

    os.makedirs("src", exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(concept_vectors, f)


    for theme_name, vector in concept_vectors.items():
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()