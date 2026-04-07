import struct
import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from collections import defaultdict
import logging

stemmer = PorterStemmer()
stemmed_stopwords = list({stemmer.stem(w) for w in ENGLISH_STOP_WORDS}) \
                     + ['anywh', 'becau', 'el', 'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh']

DOC_MAP_POSTINGS = {}
REVERSE_DOC_MAP_POSTINGS = {}
WORD_MAP_POSTINGS = {}
DOC_MAP_CATEGORIES = {}
REVERSE_DOC_MAP_CATEGORIES = {}
WORD_MAP_CATEGORIES = {}

logger = logging.getLogger(__name__)
gunicorn_logger = logging.getLogger('gunicorn.info')
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
DATA_DIR = os.path.join(project_root, 'data')

def decode_postings(blob):
    ptr = 0
    count = struct.unpack_from("I", blob, ptr)[0]
    ptr += 4

    doc = 0
    postings = []

    for _ in range(count):
        delta, score = struct.unpack_from("I H", blob, ptr)
        ptr += 6

        doc += delta
        postings.append((doc, score))

    return postings


def stem_tokenizer(text):
    words = re.findall(r"\w+", text.lower())
    return [stemmer.stem(w) for w in words]


def load_data():
    global DOC_MAP_POSTINGS, REVERSE_DOC_MAP_POSTINGS, WORD_MAP_POSTINGS
    global DOC_MAP_CATEGORIES, REVERSE_DOC_MAP_CATEGORIES, WORD_MAP_CATEGORIES
    doc_path_postings = os.path.join(DATA_DIR, "doc_map_postings.json")
    word_path_postings = os.path.join(DATA_DIR, "word_map_postings.json")
    doc_path_categories = os.path.join(DATA_DIR, "doc_map_categories.json")
    word_path_categories = os.path.join(DATA_DIR, "word_map_categories.json")

    try:
        with open(doc_path_postings, "r") as f:
            DOC_MAP_POSTINGS = json.load(f)
            REVERSE_DOC_MAP_POSTINGS = {wiki_id: title for title, wiki_id in DOC_MAP_POSTINGS.items()}

    except FileNotFoundError as e:
        print(e)
        print("Error: postings doc map not found")

    try:
        with open(word_path_postings, "r") as f:
            WORD_MAP_POSTINGS = json.load(f)
    except FileNotFoundError as e:
        print(e)
        print("Error: postings word map not found")

    try:
        with open(doc_path_categories, "r") as f:
            DOC_MAP_CATEGORIES = json.load(f)
            REVERSE_DOC_MAP_CATEGORIES = {wiki_id: title for title, wiki_id in DOC_MAP_POSTINGS.items()}
    except FileNotFoundError as e:
        print(e)
        print("Error: categories doc map not found")

    try:
        with open(word_path_categories, "r") as f:
            WORD_MAP_CATEGORIES = json.load(f)
    except FileNotFoundError as e:
        print(e)
        print("Error: categories word map not found")



def generate_rabbit_hole(start_article, additional_keywords, postings_model, path_length=5, diversity_lambda=0.5):
    """
    Returns list of articles to discover
    """

    # 1. Retrives doc using binary index
    # gunicorn.info
    global logger
    logger.info("Generating rabbit hole")
    query_text = f"{start_article} {additional_keywords}"
    tokens = stem_tokenizer(query_text)
    unique_tokens = list(set(tokens))

    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens)

    doc_scores = defaultdict(float)
    doc_vectors = defaultdict(dict)

    for token in unique_tokens:
        term_id = WORD_MAP_POSTINGS.get(token)
        if term_id is not None:
            record = postings_model.query.filter_by(term_id=term_id).first()
            if record and record.postings:
                logger.info("Found records")
                decoded = decode_postings(record.postings)

                for doc_id, score in decoded:
                    doc_scores[doc_id] += score
                    doc_vectors[doc_id][token] = score
    logger.info("Processed tokens")
            
    if not doc_scores:
        return []
    
    # 2. Applies MRR for diversity
    # 2.1: convert dict vectors to NumPy arras
    np_vectors = {}
    for doc_id, vector_dict in doc_vectors.items():
        vec = np.zeros(vocab_size)
        for term, score in vector_dict.items():
            if term in token_to_idx:
                vec[token_to_idx[term]] = score
        np_vectors[doc_id] = vec

    # 2.2: Do MMR
    candidates = list(doc_scores.keys())
    # print(candidates)
    # print(len(candidates))
    pathway = []
    
    for _ in range(path_length):
        if not candidates:
            break
    
        best_doc = None
        best_mmr_score = -float('inf')

        for doc in candidates:
            relevance = doc_scores[doc]
            similarity_penalty = 0
            if pathway:
                penalties = []
                for selected_doc in pathway:
                    sim = np.dot(np_vectors[doc], np_vectors[selected_doc])
                    penalties.append(sim)
                similarity_penalty = max(penalties)

            mmr_score = (diversity_lambda * relevance) - ((1-diversity_lambda) * similarity_penalty)
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_doc = doc
            
        if best_doc is not None:
            pathway.append(best_doc)
            candidates.remove(best_doc)
    
    # 3. Format output
    res = []
    for doc_id in pathway:
        title = REVERSE_DOC_MAP_POSTINGS.get(doc_id, f"Unknown ID {doc_id}")
        res.append({
            "id": doc_id,        # already IS the wiki ID
            "title": title,
            "score": round(doc_scores[doc_id], 2)
        })
        print(doc_id)
    return res


def generate_rabbit_hole_2(start_article, postings_model, categories_model, path_length=5, alpha=0.67):
    tokens = stem_tokenizer(start_article)
    unique_tokens = list(set(tokens))

    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens)

    postings_scores = defaultdict(float)
    postings_token_counts = defaultdict(int)

    categories_scores = defaultdict(float)
    categories_token_counts = defaultdict(int)

    for token in unique_tokens:
        term_id_posting = WORD_MAP_POSTINGS.get(token)
        term_id_category = WORD_MAP_CATEGORIES.get(token)
        if term_id_posting:
            record = postings_model.query.filter_by(term_id=term_id_posting).first()
            if record and record.postings:
                logger.info("Found records")
                decoded = decode_postings(record.postings)

                for doc_id, score in decoded:
                    postings_scores[doc_id] += score
                    postings_token_counts[doc_id] += 1
        if term_id_category:
            record = categories_model.query.filter_by(term_id=term_id_category).first()
            if record and record.postings:
                logger.info("Found records")
                decoded = decode_postings(record.postings)

                for doc_id, score in decoded:
                    categories_scores[doc_id] += score
                    categories_token_counts[doc_id] += 1

    final_scores = defaultdict(float)
    for doc_id, score in postings_scores.items():
        title = REVERSE_DOC_MAP_POSTINGS.get(doc_id, f"Unknown ID {doc_id}")
        final_scores[title] += score * alpha * (postings_token_counts[doc_id] ** 2)

    for doc_id, score in categories_scores.items():
        title = REVERSE_DOC_MAP_CATEGORIES.get(doc_id, f"Unknown ID {doc_id}")
        final_scores[title] += score * (1 - alpha) * (categories_token_counts[doc_id] ** 2)

    sorted_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    res = []
    print(sorted_ids[:100])
    i = 0
    while len(res) < path_length:
        title, score = sorted_ids[i]
        doc_id = DOC_MAP_POSTINGS.get(title, f"Unknown Article {title}")
        if not title.startswith("Unknown ID"):
            res.append({
                "id": doc_id,        # already IS the wiki ID
                "title": title,
                "score": round(score, 2)
            })
        i += 1
    return res

