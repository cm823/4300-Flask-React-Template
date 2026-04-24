import struct
import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from collections import defaultdict
from models import Articles
import logging
import random
vecdot = np.vdot

non_people_articles = {
    "Alligatoridae"
}

stemmer = PorterStemmer()
stemmed_stopwords = set(list({stemmer.stem(w) for w in ENGLISH_STOP_WORDS}) \
                     + ['anywh', 'becau', 'el', 'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh'])

dimension_themes = {
    0:  "Film & Television 1",
    1:  "Film & Television 2",
    2:  "Music 1",
    3:  "Music 2",
    4:  "Film & Television 3",
    5:  "Basketball",
    6:  "Football & Team Sports 1",
    7:  "Baseball",
    8:  "Politics & Government 1",
    9:  "Film & Television 3",
    10: "Combat Sports & Wrestling 1",
    11: "Ice Hockey 1",
    12: "Film & Television 4",
    13: "Ice Hockey 2",
    14: "Music 3",
    15: "Indian Cinema & South Asian 1",
    16: "Broadcasting & Journalism 1",
    17: "Football & Team Sports 2",
    18: "Literature & Academia 1",
    19: "Geography – New York 1",
    20: "Combat Sports & Wrestling 2",
    21: "Football & Team Sports 3",
    22: "Broadcasting & Journalism 2",
    23: "Literature & Academia 2",
    24: "Politics & Government 2",
    25: "Cricket",
    26: "Film & Television 5",
    27: "Film & Television 6",
    28: "Music 4",
    29: "Geography – New York 2",
    30: "Crime & Legal",
    31: "Motor Racing",
    32: "Broadcasting & Journalism 3",
    33: "Business & Corporate 1",
    34: "Fashion & Modelling 1",
    35: "Indian Cinema & South Asian 2",
    36: "Olympics & Athletics 1",
    37: "Politics & Government 3",
    38: "Australian Sports & Rugby 1",
    39: "Fashion & Modelling 2",
    40: "Politics & Government 4",
    41: "Broadcasting & Journalism 4",
    42: "Music 5",
    43: "Animation, Gaming & Comics 1",
    44: "Music 6",
    45: "Theatre & Stage 1",
    46: "Animation, Gaming & Comics 2",
    47: "Australian Sports & Rugby 2",
    48: "Animation, Gaming & Comics 3",
    49: "Olympics & Athletics 2",
    50: "Nationality – British 1",
    51: "Nationality – British 2",
    52: "Nationality – British 3",
    53: "Literature & Academia 3",
    54: "Nationality – North American 1",
    55: "Animation, Gaming & Comics 4",
    56: "Indian Cinema & South Asian 3",
    57: "Music 7",
    58: "Nationality – Central European 1",
    59: "Nationality – North American 2",
    60: "Nationality – French 1",
    61: "Indian Cinema & South Asian 4",
    62: "Nationality – French/Canadian",
    63: "Nationality – North American 3",
    64: "Nationality – British 4",
    65: "Indian Cinema & South Asian 5",
    66: "Nationality – North American 4",
    67: "Animation, Gaming & Comics 5",
    68: "Nationality – North American 5",
    69: "Theatre & Stage 2",
    70: "Nationality – South/Japanese",
    71: "Broadcasting & Journalism 5",
    72: "Geography – Western United States 1",
    73: "Nationality – British 5",
    74: "Geography – Western United States 2",
    75: "Nationality – Australian/Japanese",
    76: "Nationality – Japanese 1",
    77: "Theatre & Stage 3",
    78: "Nationality – Japanese/German",
    79: "Nationality – Western Europe",
    80: "Nationality – South African",
    81: "Business & Corporate 2",
    82: "Nationality – French 2",
    83: "Nationality – Japanese/Canadian",
    84: "Theatre & Stage 4",
    85: "Boxing",
    86: "Nationality – Japanese 2",
    87: "Broadcasting & Journalism 6",
    88: "Nationality – French/Chinese",
    89: "Nationality – Central European 2",
    90: "Nationality – New Zealand",
    91: "Music 8",
    92: "Golf",
    93: "Literature & Academia 4",
    94: "Middle East & Iran",
    95: "Nationality – Russian/Soviet",
    96: "Nationality – Irish",
    97: "Nationality – Italian",
    98: "Nationality – Central European 3",
    99: "Nationality – Centran European 4",
}

DOC_MAP = {}
REVERSE_DOC_MAP = {}
WORD_MAP = {}
WORD_ID_TO_TERM = {}
DOC_EMBEDDINGS = np.zeros(1)
TERM_EMBEDDINGS = np.zeros(1)
SINGULAR_VALUES = np.zeros(1)
DOC_IDS_SVD = {}
DOC_IDS_SVD_REVERSE = {}

logger = logging.getLogger(__name__)
gunicorn_logger = logging.getLogger('gunicorn.info')
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)

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
    words = [word for word in words if word not in stemmed_stopwords]
    n_grams = []
    for i in range(2, 4):
        for j in range(len(words)-i+1):
            n_grams.append(" ".join(words[j:j+i]))
    return [stemmer.stem(w) for w in words + n_grams]

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
DATA_DIR = os.path.join(project_root, 'data')

def load_data():
    global DOC_MAP, REVERSE_DOC_MAP, WORD_MAP, WORD_ID_TO_TERM, DOC_EMBEDDINGS, TERM_EMBEDDINGS, DOC_IDS_SVD, DOC_IDS_SVD_REVERSE, SINGULAR_VALUES
    doc_path = os.path.join(DATA_DIR, "doc_map3.json")
    word_path = os.path.join(DATA_DIR, "word_map3.json")
    doc_embeddings_path_1 = os.path.join(DATA_DIR, "svd_scipy4", "doc_embeddings1.npy")
    doc_embeddings_path_2 = os.path.join(DATA_DIR, "svd_scipy4", "doc_embeddings2.npy")
    term_embeddings_path = os.path.join(DATA_DIR, "svd_scipy4", "term_embeddings.npy")
    doc_ids_path = os.path.join(DATA_DIR, "svd_scipy4", "doc_ids.txt")
    singular_values_path = os.path.join(DATA_DIR, "svd_scipy4", "singular_values.npy")

    try:
        with open(doc_path, "r") as f:
            DOC_MAP = json.load(f)
            # REVERSE_DOC_MAP = {idx: title for idx, title in enumerate(DOC_MAP.keys())}
            # REVERSE_DOC_MAP = [(v, k) for k, v in DOC_MAP.items()]
            REVERSE_DOC_MAP = {wiki_id: title for title, wiki_id in DOC_MAP.items()}

    except FileNotFoundError as e:
        print(e)
        print("Error: boi pls put doc_map.json in <root>/data")

    try:
        with open(word_path, "r") as f:
            WORD_MAP = json.load(f)
            WORD_ID_TO_TERM = {v: k for k, v in WORD_MAP.items()}
    except FileNotFoundError as e:
        print(e)
        print("Error: pls have the world_map.json in the data folder!")

    arr1 = np.load(doc_embeddings_path_1, allow_pickle=True)
    arr2 = np.load(doc_embeddings_path_2, allow_pickle=True)
    DOC_EMBEDDINGS = np.concatenate([arr1, arr2], axis=0)
    TERM_EMBEDDINGS = np.load(term_embeddings_path)
    SINGULAR_VALUES = np.load(singular_values_path)


    with open(doc_ids_path) as f:
        for line in f:
            i, name = line.strip().split("\t", 1)
            DOC_IDS_SVD[int(i)] = name
            DOC_IDS_SVD_REVERSE[name] = int(i)

def get_svd_graph_data(terms_per_theme=8):
    """
    Returns SNAP-format graph data (nodes + edges) grouped by dimension_themes.
    Each unique theme becomes a cluster; terms are aggregated across all dims
    sharing that theme. Edges connect terms within the same theme cluster.
    """
    theme_to_dims = defaultdict(list)
    for dim, theme in dimension_themes.items():
        if dim < TERM_EMBEDDINGS.shape[1]:
            theme_to_dims[theme].append(dim)

    unique_themes = sorted(theme_to_dims.keys())
    theme_to_idx = {t: i for i, t in enumerate(unique_themes)}

    nodes = {}   # term -> node dict
    edges = []

    for theme, dims in theme_to_dims.items():
        cluster_idx = theme_to_idx[theme]
        term_scores = defaultdict(float)

        for dim in dims:
            col = TERM_EMBEDDINGS[:, dim]
            top_idx = np.argsort(col)[::-1][: terms_per_theme * 3]
            for i in top_idx:
                term = WORD_ID_TO_TERM.get(int(i))
                if term:
                    term_scores[term] += float(col[i])

        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:terms_per_theme]

        cluster_ids = []
        for term, score in top_terms:
            if term not in nodes:
                nodes[term] = {
                    "id": term,
                    "label": term,
                    "cluster": cluster_idx,
                    "theme": theme,
                    "weight": round(score, 4),
                }
            cluster_ids.append(term)

        # Edges: fully connect terms within the same theme cluster
        for a in range(len(cluster_ids)):
            for b in range(a + 1, len(cluster_ids)):
                edges.append({"source": cluster_ids[a], "target": cluster_ids[b]})

    return {"nodes": list(nodes.values()), "edges": edges, "themes": unique_themes}

def generate_rabbit_hole(start_article, additional_keywords, postings_model, path_length=5, diversity_lambda=0.5, num_branches=3, branch_seeds=None, randomize=True):
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
        term_id = WORD_MAP.get(token)
        if term_id is not None:
            record = postings_model.query.filter_by(term_id=term_id).first()
            if record and record.postings:
                logger.info("Found records")
                decoded = decode_postings(record.postings)

                for doc_id, score in decoded:
                    score /= 10000
                    doc_scores[doc_id] += score
                    doc_vectors[doc_id][token] = score
    logger.info("Processed tokens")
    print("Processed tokens")
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
    print("Did MMR")
    # 2.2: Do MMR
    candidates = list(doc_scores.keys())
    pathway = []
    
    for _ in range(path_length * num_branches * 2):
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
        # print(f"retrieved candidates {_}")
            
        if best_doc is not None:
            pathway.append(best_doc)
            candidates.remove(best_doc)
    
    # 3. Format output 
    # Changed to branch nodes as frontend expects many branch nodes for each rabbit hole. 
    branch_nodes = []

    description = "Your person to explore!"

    if randomize:
        np.random.shuffle(pathway)

    for i in range(0, path_length*num_branches, path_length):
        nodes = pathway[i:i+path_length]
        temp = []
        for doc_id in nodes:
            if doc_id not in REVERSE_DOC_MAP:
                continue
            title = REVERSE_DOC_MAP.get(doc_id, f"Unknown ID {doc_id}")
            try:
                text = Articles.query.filter_by(article_name=title).first().article_text
            except Exception as e:
                text = ""
            if title.startswith("Unknown ID") or title in non_people_articles:
                continue
            temp.append({
                "id": doc_id,
                "title": title,
                "score": round(doc_scores[doc_id], 4),
                "branch": int(i/path_length) + 1,
                "description": description,
                "text" : text
            })

        if temp:
            branch_nodes.append(temp)

    return branch_nodes


def generate_branch(doc_id, doc_vec, doc_embed, term_embed, path_length=5):
    path = [doc_id]
    prev_doc = doc_vec
    j = 0
    while len(path) < path_length and j < 20:
        q_emb = prev_doc @ term_embed
        q_emb /= np.linalg.norm(q_emb) + 1e-8
        scores = doc_embed @ q_emb
        top_idx = np.argsort(scores)[-5:][::-1]
        np.random.shuffle(top_idx)

        i = 0
        if top_idx[0] in path:
            i += 1

        path.append(top_idx[i])
        prev_doc = TERM_EMBEDDINGS @ (SINGULAR_VALUES * DOC_EMBEDDINGS[top_idx[i], :])
        j += 1

    return path

def top_query_dimensions(q_emb, top_k=5):
    # sort by magnitude (important: absolute value)
    idx = np.argsort(np.abs(q_emb))[::-1][:top_k]

    return [(i, q_emb[i]) for i in idx]

def gen_query_vec(query):
    tokens = stem_tokenizer(query)
    unique_tokens = list(set(tokens))

    num_terms = TERM_EMBEDDINGS.shape[0]
    vec = np.zeros(num_terms, dtype=np.float32)
    print(vec.shape)

    for w in unique_tokens:
        if w in WORD_MAP:
            vec[WORD_MAP[w]] += 1.0

    return vec

def generate_rabbit_hole_svd(start_article, path_length=5, num_branches=3):
    global TERM_EMBEDDINGS, DOC_EMBEDDINGS, DOC_IDS_SVD, SINGULAR_VALUES

    vec = gen_query_vec(start_article)

    q_emb = vec @ TERM_EMBEDDINGS

    q_emb /= np.linalg.norm(q_emb) + 1e-8

    scores = DOC_EMBEDDINGS @ q_emb

    top_idx = np.argsort(scores)[-(path_length*num_branches):][::-1]

    np.random.shuffle(top_idx)

    branch_nodes = []

    description = "A unique thematic cluster."

    for i in range(0, path_length * num_branches, path_length):
        nodes = top_idx
        temp = []
        top_q_dims = top_query_dimensions(q_emb, top_k=6)
        dim_names = []
        dim_scores = []
        for dim, score in top_q_dims:
            if dimension_themes[dim] not in dim_names:
                dim_names.append(dimension_themes[dim])
                dim_scores.append(float(score))
        temp.append({
            "id" : -1,
            "title" : "QUERY",
            "score" : -1,
            "branch" : 0,
            "description" : description,
            "dimensions" : dim_names,
            "dimensionScores" : dim_scores,
            "text" : ""
        })
        for node in nodes:
            doc_vec = TERM_EMBEDDINGS @ (SINGULAR_VALUES * DOC_EMBEDDINGS[node, :])
            dims = top_query_dimensions(doc_vec @ TERM_EMBEDDINGS, top_k=100)
            dims.sort(key=lambda x: x[0])
            dim_names = []
            dim_scores = []
            title = DOC_IDS_SVD[node]
            try:
                text = Articles.query.filter_by(article_name=title).first().article_text
            except Exception as e:
                text = ""
            for dim, score in top_q_dims:
                dim_names.append(dimension_themes[dim])
                dim_scores.append(float(dims[dim][1]))
            if title in non_people_articles:
                continue
            temp.append(
                {
                    "id": int(node),
                    "title": title,
                    "score": round(float(scores[node]), 4),
                    "branch": int(i/path_length)+1,
                    "description": description,
                    "dimensions": dim_names,
                    "dimensionScores" : dim_scores,
                    "text" : text
                }
            )
            if len(temp) >= path_length + 1:
                break
        branch_nodes.append(temp)

    return branch_nodes

def minmax(x):
    if len(x) == 0:
        return x  # or np.array([])

    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def generate_rabbit_hole_combined(start_article, additional_keywords, postings_model,  path_length=5, num_branches=3):
    global TERM_EMBEDDINGS, DOC_EMBEDDINGS, DOC_IDS_SVD, SINGULAR_VALUES

    vec = gen_query_vec(start_article) @ TERM_EMBEDDINGS

    vec /= np.linalg.norm(vec) + 1e-8

    top_q_dims = top_query_dimensions(vec, top_k=6)
    dim_names = []
    dim_scores = []
    for dim, score in top_q_dims:
        if dimension_themes[dim] not in dim_names:
            dim_names.append(dimension_themes[dim])
            dim_scores.append(float(score))

    tokens = stem_tokenizer(start_article)
    unique_tokens = list(set(tokens))

    doc_scores = defaultdict(float)

    for token in unique_tokens:
        print(token)
        term_id = WORD_MAP.get(token)
        if term_id is not None:
            record = postings_model.query.filter_by(term_id=term_id).first()
            if record and record.postings:
                logger.info("Found records")
                decoded = decode_postings(record.postings)

                for doc_id, score in decoded:
                    score /= 10000
                    doc_scores[doc_id] += score
    print("done processing tokens")
    print(len(doc_scores))
    candidate_pool = max(path_length * 20, 200)
    sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool]
    print(len(sorted_scores))

    doc_cos_results = []

    description = "A unique thematic cluster."

    for doc_id, score in sorted_scores:
        if doc_id not in REVERSE_DOC_MAP:
            continue
        title = REVERSE_DOC_MAP.get(doc_id, f"Unknown ID {doc_id}")
        if title.startswith("Unknown ID"):
            continue
        if title in non_people_articles:
            continue
        # skip docs not present in the SVD vocab
        if title not in DOC_IDS_SVD_REVERSE:
            continue
        try:
            text = Articles.query.filter_by(article_name=title).first().article_text
        except Exception as e:
            text = ""
        doc_cos_results.append({
            "id" : doc_id,
            "title" : title,
            "score" : round(float(score), 4),
            "branch" : 0,
            "description": description,
            "text": text
        })
    print("done jsonifying")

    if not doc_cos_results:
        return []

    doc_cos_results.sort(key=lambda x: x["score"], reverse=True)

    doc_idxs = []

    for doc in doc_cos_results:
        svd_id = DOC_IDS_SVD_REVERSE[doc['title']]
        doc["id"] = svd_id
        doc_idxs.append(svd_id)

    doc_vecs = DOC_EMBEDDINGS[doc_idxs]

    svd_scores = doc_vecs @ vec.T
    cos_scores = np.array([doc['score'] for doc in doc_cos_results])

    cos_scores = minmax(cos_scores)
    svd_scores = minmax(svd_scores)

    print("done svding")

    final_scores = 0.8 * cos_scores + 0.2 * svd_scores

    for i in range(len(doc_cos_results)):
        doc_cos_results[i]["score"] = round(float(final_scores[i]), 4)

    doc_cos_results.sort(key=lambda x: x["score"], reverse=True)

    final_results = doc_cos_results[:path_length*2]

    random.shuffle(final_results)

    final_results.insert(0, {
            "id" : -1,
            "title" : "QUERY",
            "score" : -1,
            "branch" : 0,
            "description" : description,
            "dimensions" : dim_names,
            "dimensionScores" : dim_scores,
            "text" : ""
        })

    for result in final_results[1:path_length+1]:
        doc_vec = TERM_EMBEDDINGS @ (SINGULAR_VALUES * DOC_EMBEDDINGS[result["id"], :])
        dims = top_query_dimensions(doc_vec @ TERM_EMBEDDINGS, top_k=100)
        dims.sort(key=lambda x: x[0])
        dim_names = []
        dim_scores = []
        for dim, score in top_q_dims:
            dim_names.append(dimension_themes[dim])
            dim_scores.append(float(dims[dim][1]))
        result["dimensions"] = dim_names
        result["dimensionScores"] = dim_scores
        print(result)


    return [final_results[:path_length+1]]



