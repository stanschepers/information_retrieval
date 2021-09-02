import math
import pickle
import time
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasketch import MinHash, MinHashLSH
from datasketch.lsh import MinHashLSHInsertionSession

from Assignments.report2.preprocess_nlp import preprocess_nlp


def jaccard_similarity(doc1, doc2):
    """ Jaccard similarity between 2 documents (set of words) """
    n_intersection = len(doc1.intersection(doc2))  # Number of words that are both in documents
    n_union = len(doc1.union(doc2))  # Number of different words in both documents
    n_intersection, n_union = float(n_intersection), float(n_union)  # Make them float to have float division
    return n_intersection / n_union


def preprocess_nlp_set(id_text, stages):
    """  Preprocess text to make a set of words of a document """
    news_id, text = id_text
    processed_text = set(preprocess_nlp(text, stages))
    return news_id, processed_text


def shingle(l, k):
    return [" ".join(l[i: i + k]) for i in range(len(l) - k + 1)]


def preprocess_shingle(id_text, k=2):
    news_id, text = id_text
    stages = {'remove_tags_nonascii': True,
              'lower_case': True,
              'expand_contractions': False,
              'remove_escape_chars': True,
              'remove_punctuation': True,
              'remove_stopwords': False,
              'remove_numbers': False,
              'lemmatize': False,
              'stemming': False,
              'min_word_len': 1
              }
    text = preprocess_nlp(text, stages)  # only tokenise, remove non_ascii and escape chars, lowercase every thing
    processed_text = set(shingle(text, k))
    return news_id, processed_text


def evaluate_docs(doc_ids, data):
    """ Evaluate docs using their ids from a data dict using jaccard similarity"""
    doc_id1, doc_id2 = doc_ids
    return doc_id1, doc_id2, jaccard_similarity(data[doc_id1], data[doc_id2])


def min_hash(doc, n_permutations=128):
    """ Make MinHash from document (set of words) with n_permutations permutations """
    doc_id, doc_set = doc
    m = MinHash(num_perm=n_permutations)
    m.update_batch([d.encode('utf-8') for d in doc_set])
    return doc_id, m


def minhash_matches(threshold, n_permutations, r=None, b=None, weights=None, use_multicore=False, rbs=None):
    """ """

    """ Create MinHashes for documents """
    if rbs is None:
        rbs = list()
    if use_multicore:
        with Pool(processes=6) as pool:
            minhashes = pool.map(partial(min_hash, n_permutations=n_permutations), data)
    else:
        minhashes = [min_hash(d, n_permutations=n_permutations) for d in data]

    """ Create Similarity Matrix """
    if r is not None or b is not None:
        params = (r, b)
        lsh = MinHashLSH(threshold=threshold, num_perm=n_permutations, params=params)
    elif weights is not None:
        lsh = MinHashLSH(threshold=threshold, num_perm=n_permutations, weights=weights)
    else:
        lsh = MinHashLSH(threshold=threshold, num_perm=n_permutations)

    if (lsh.b, lsh.r) in rbs:
        return


    with MinHashLSHInsertionSession(lsh, 1000) as sess:
        for doc_id, m in minhashes:
            lsh.insert(str(doc_id), m)

    """ Get Matches (similarity >= threshold) """
    minhash_matches = set()
    for doc_id, m in minhashes:
        doc_id = str(doc_id)
        results = lsh.query(m)
        for result in results:
            if result != doc_id:
                match = sorted((int(result), int(doc_id)))
                minhash_matches.add(tuple(match))
    return list(minhash_matches), lsh


if __name__ == '__main__':

    """ Read in articles and preprocess the strings and make from a document a set of words."""

    dataset = "news_articles_large"
    threshold = 0.8
    MULTICORE = True
    score = False
    preprocessing_k = 0  # 0 for nlp else k shingles with k = preprocessing_k

    df = pd.read_csv(dataset + ".csv", index_col="News_ID")
    ids = df.index.tolist()
    tuples = [tuple(x) for x in df.itertuples()]

    print("Dataset:", dataset)
    print("Threshold:", threshold)
    print("Using multicore:", MULTICORE)
    print("Preprocessing and Calculating Scores:", score)
    print("Type of preprocessing:", preprocessing_k)

    stages = dict(remove_stopwords=True, stemming=False)

    """Preprocessing  """

    start_time = time.time()
    if score:
        with Pool(processes=6) as pool:
            if preprocessing_k > 0:
                preprocess_fn = partial(preprocess_shingle, k=preprocessing_k)
            else:
                preprocess_fn = partial(preprocess_nlp_set, stages=stages)
            data = pool.map(preprocess_fn, tuples)
        with open(f"{dataset}_{preprocessing_k}_data.pickle", "wb+") as file:
            pickle.dump(data, file)
    else:
        with open(f"{dataset}_{preprocessing_k}_data.pickle", "rb+") as file:
            data = pickle.load(file)

    print("End time:", round(time.time() - start_time, 4))
    average = lambda x: sum(x) / len(x)
    print("Average document size:", average([len(d[1]) for d in data]))
    print([d[1] for d in data[:2]])

    """ Evaluate all documents with each other and save the results (scores) in a CSV file """
    comb = combinations(ids, 2)
    n_combinations = math.comb(len(data), 2)

    if score:
        data_dict = dict(data)

        print("Start Brute Force")
        bruteforce_start_time = time.time()

        if MULTICORE:
            with Pool(processes=6) as pool:
                scores = pool.map(partial(evaluate_docs, data=data_dict), comb)
        else:
            scores = [evaluate_docs(c, data=data_dict) for c in comb]

        print("Brute force time (in s):", int(time.time() - bruteforce_start_time))

    """ Read or Store Score Matches """
    score_filename = dataset + "_scores.csv"
    if score:
        print("Storing score file:", score_filename)
        df_scores = pd.DataFrame(scores, columns=['doc_id1', 'doc_id2', 'score'])
        df_scores.to_csv(score_filename, index=False)
    else:
        print("Reading score file:", score_filename)
        df_scores = pd.read_csv(dataset + "_scores.csv", )
    score_matches = list(df_scores[df_scores["score"] >= threshold][["doc_id1", "doc_id2"]].values)
    score_matches = [(x, y) if x < y else (y, x) for (x, y) in score_matches]

    print("# Score matches:", len(score_matches))

    """ Sava a histogram from the jaccard similarity with the different documents """

    fig, ax1 = plt.subplots()

    ax1.hist(df_scores["score"], bins=10, color="pink")
    ax1.set_yscale('log')
    ax1.set_xlabel("Jaccard index")
    ax1.set_ylabel("# document pairs")

    ax2 = ax1.twinx()

    def ffp(similarity, r, b):
        return 1 - (1 - similarity**r)**b
    br_pairs = [(1, 2), (2, 3), (2, 7), (4, 4), (6, 5), (4, 8), (32, 1), (11, 11, ),  (1, 128)]
    similarities = np.arange(0, 1, 0.01)
    for (b, r) in br_pairs:
        ax2.plot(similarities, [ffp(s, r, b) for s in similarities], label=str((b, r)))
    ax2.set_ylim(bottom=0)
    ax2.vlines(0.8, ymin=0.05, ymax=1, label="threshold 0.8", color="crimson")
    ax2.vlines(0.9, ymin=0.05, ymax=1, label="threshold 0.9", color="gold")
    ax2.set_ylabel("Prob. sharing bucket")
    ax1.set_title(f"Histogram: {dataset} Preprocessing: {f'{preprocessing_k}_shingle' if preprocessing_k > 0 else 'NLP'}")
    ax2.legend()
    fig.savefig(f"{dataset}_{preprocessing_k}_scores.png")
    fig.tight_layout()
    plt.show()

    n_permutations_list = [2, 4, 8, 16, 32, 64, 128]
    false_positive_weights = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]

    # mh_matches, lsh = minhash_matches(threshold, n_permutations=128, use_multicore=MULTICORE)

    results = list()

    for n_permutations in n_permutations_list:
        n_fp_mismatches = list()
        n_fn_mismatches = list()
        for fpw in false_positive_weights:
            fnw = 1 - fpw
            print("Start MinHashing LSH with # permutations:", n_permutations)
            minhash_start_time = time.time()
            result = minhash_matches(threshold, n_permutations, weights=(fpw, fnw), use_multicore=MULTICORE)
            if results is None:
                continue
            mh_matches, lsh = result
            end_time = round(time.time() - minhash_start_time, 3)
            print("Minhash time (in s) for # permutations", n_permutations, ":", end_time)
            print("Size of LSH:", lsh.h)

            print("n permutations:", n_permutations)
            print("r:", lsh.r, " b:", lsh.b)
            match_scores_minhashes = set(score_matches) == set(mh_matches)

            print("# matches:", len(mh_matches), "match with calculated scores:", match_scores_minhashes)
            fp_mismatches = [m for m in mh_matches if m not in score_matches]
            fn_mismatches = [m for m in score_matches if m not in mh_matches]
            tp_matches = [m for m in mh_matches if m in score_matches]
            n_fp_mismatches.append(len(fp_mismatches))
            n_fn_mismatches.append(len(fn_mismatches))

            tp, fn, fp, tn = float(len(tp_matches)), float(len(fn_mismatches)), float(len(fp_mismatches)), float(
                n_combinations - len(score_matches))
            precision, recall, specificity = tp / (tp + fp), tp / float(len(score_matches)), tn/float(n_combinations)
            s = (1/lsh.b)**(1/lsh.r)
            results.append((n_permutations, lsh.b, lsh.r, recall, specificity, precision, end_time, s))
            print(results[-1])
    results = pd.DataFrame(list(set(results)), columns=["M", "b", "r", "sensitivity/recall", "specificity", "precision", "time (in s)", "aprrox. threshold"])

    print(results.drop_duplicates(subset=["M", "b", "r"]).sort_values(["M", "b", "r"]).to_markdown(index=False))


    #
    #     fig, ax1 = plt.subplots()
    #
    #     ax1.bar(false_positive_weights, n_fp_mismatches, color="blue", width=0.05)
    #     ax1.set_yscale("log")
    #
    #     ax2 = ax1.twinx()
    #
    #     ax2.bar(false_positive_weights, n_fn_mismatches, color="red", width=0.05)
    #
    #     ax1.set_xlabel("False Positive Weight")
    #     ax1.set_ylabel("# false positive mismatches (blue)")
    #     ax2.set_ylabel("# false negative mismatches (red)")
    #     ax1.set_title(f"n permutations: {n_permutations}")
    #     plt.show()

    # if len(mismatches) > 0:
    #     print("Matches not in score matches:")
    #     for m in mismatches:
    #         print(df.loc[int(m[0])])
    #         print(df.loc[int(m[1])])
    #         print("—————————————————————————")
    #
    # """ Save MinHash Matches """
    # df_mh_matches = pd.DataFrame(mh_matches, columns=["doc_id1", "doc_id2"])
    # df_mh_matches.to_csv(dataset + f"_{n_permutations}_matches.csv", index=False)
