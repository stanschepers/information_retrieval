"""
Experiments


experimetns

"""
import datetime
import multiprocessing
import random
import time
from multiprocessing import Pool

import pandas as pd

from Assignments.report2.topic_modeling import preprocess_texts, build_lda_model, evaluate_lda, get_first_n_sentences

DEFAULT_N_TOPICS = [10, 50]

N_EXPERIMENTS = 5

N_WORKERS = 5


def calculate_statitics_results(results, parameters):
    columns = parameters + ["coherence"]
    results_df = pd.DataFrame(results, columns=columns)
    mean, std = results_df.groupby(parameters).mean(), results_df.groupby(parameters).std()
    mean, std = mean.reset_index(), std.reset_index()
    statistics = pd.merge(mean.rename(columns=dict(coherence="coherence_mean")),
                          std.rename(columns=dict(coherence="coherence_std")), on=parameters)
    return statistics


def chunks(lst, n):
    """ from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _fail_safe_experiment(fn):
    try:
        return fn
    except Exception as e:
        print(e)


def experiment(name, data, fn, parameters):
    start_time = time.time()
    print("Starting on", name)

    try:
        with Pool(N_WORKERS) as pool:
            results_list = pool.map(fn, chunks(data, len(data) // N_EXPERIMENTS))
        results = []
        for result in results_list:
            results.extend(result)
        statistics = calculate_statitics_results(results, parameters=parameters)
        statistics.to_csv(f"experiment_{name}_results.csv")
        print("Results", name)
        print(statistics.to_markdown(index=False))
    except Exception as e:
        print("Exception:", e, "at", name)
    end_time_m = (time.time() - start_time) / 60
    print("Time (in min.):", round(end_time_m))


def _experiment_n_sentences(data):
    results = list()

    n_sentences_list = [10, 50, 100, -1]

    for n_topics in [10]:
        for n_sentences in n_sentences_list:
            data_n_sentences = get_first_n_sentences(data, n_sentences)
            dictionary, corpus, texts = preprocess_texts(data_n_sentences, {})
            model = build_lda_model(corpus, dictionary, n_topics=n_topics)
            if model is not None:
                perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
            else:
                coherence = -1
            results.append((n_topics, n_sentences, coherence))
    return results


def _experiment_lemmatize_vs_stemming(data):
    results = list()

    stem_config = dict(stemming=True, lemmatize=False)
    lemm_config = dict(stemming=False, lemmatize=True)

    for n_topics in DEFAULT_N_TOPICS:
        dictionary_stem, corpus_stem, texts_stem = preprocess_texts(data, stem_config)
        stem_model = build_lda_model(corpus_stem, dictionary_stem, n_topics=n_topics)
        if stem_model is not None:
            _, coherence_stem = evaluate_lda(stem_model, dictionary_stem, corpus_stem, texts_stem)
        else:
            coherence_stem = -1
        dictionary_lemm, corpus_lemm, texts_lemm = preprocess_texts(data, lemm_config)
        lemm_model = build_lda_model(corpus_lemm, dictionary_lemm, n_topics=n_topics)
        if lemm_model is not None:
            _, coherence_lemm = evaluate_lda(lemm_model, dictionary_lemm, corpus_lemm, texts_lemm)
        else:
            coherence_lemm = -1
        results.append((n_topics, "stemming", coherence_stem))
        results.append((n_topics, "lemmatization", coherence_lemm))
    return results


def _experiment_ngrams(data):
    results = list()

    ngrams_list = [1, 2, 3]
    default_config = dict()

    for n_topics in DEFAULT_N_TOPICS:
        for ngrams in ngrams_list:
            dictionary, corpus, texts = preprocess_texts(data, default_config, ngram=ngrams)
            model = build_lda_model(corpus, dictionary, n_topics=n_topics)
            if model is not None:
                perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
            else:
                coherence = -1
            results.append((n_topics, ngrams, coherence))
    return results


def _experiment_filtering_extremes(data):
    results = list()

    above_list = [0.4, 0.5, 0.75]
    below_list = [5, 10, 20, 25]
    default_config = dict()

    # main_dictionary, main_corpus, main_texts = preprocess_texts(data, default_config, no_above=1, no_below=1)

    for n_topics in [10]:
        for above in above_list:
            for below in below_list:
                dictionary, corpus, texts = preprocess_texts(data, {}, no_above=above, no_below=below)
                model = build_lda_model(corpus, dictionary, n_topics=n_topics)
                if model is not None:
                    perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
                else:
                    coherence = -1
                results.append((n_topics, above, below, coherence))
    return results


def _experiment_n_passes(data):
    results = list()
    dictionary, corpus, texts = data

    n_passes_list = [1, 5, 10, 20, 50, 100]

    for n_topics in DEFAULT_N_TOPICS:
        for n_passes in n_passes_list:
            model = build_lda_model(corpus, dictionary, n_topics=n_topics, passes=n_passes)
            if model is not None:
                perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
            else:
                coherence = -1
            results.append((n_topics, n_passes, coherence))
    return results


def _experiment_n_passes(data):
    results = list()
    dictionary, corpus, texts = preprocess_texts(data, {})

    n_passes_list = [1, 5, 10, 20, 50, 100, 200]

    for n_topics in [10]:
        for n_passes in n_passes_list:
            model = build_lda_model(corpus, dictionary, n_topics=n_topics, passes=n_passes)
            if model is not None:
                perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
            else:
                coherence = -1
            results.append((n_topics, n_passes, coherence))
    return results


def experiment_n_topics(data):
    """ Special case """
    results = list()
    n_topic_list = [5, 10, 20, 30, 40, 50, 100]

    dictionary, corpus, texts = preprocess_texts(data, {})

    for n_topics in n_topic_list:
        model = build_lda_model(corpus, dictionary, n_topics=n_topics, use_multicore=True)
        # Fail Safe
        if model is not None:
            perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)
        else:
            coherence = -1
        results.append((n_topics, coherence))
    df = pd.DataFrame(results, columns=["n_topics", "coherence"])
    mean = df.groupby(["n_topics"]).mean()
    mean.to_csv("n_topics_results.csv")
    print("n_topics")
    print(mean)


if __name__ == '__main__':
    data = pd.read_csv("news_dataset.csv")["content"].sample(frac=0.5)
    data = data.dropna().values.tolist()
    random.shuffle(data)

    # Parameters
    print("Number of experiments:", N_EXPERIMENTS)
    print("Number of processes:", N_WORKERS)
    print("N topics:", DEFAULT_N_TOPICS)
    print("Starting on:", datetime.datetime.now())

    # Preprocessing
    # experiment("n_sentences", data, _experiment_n_sentences, ["n_topics", "n_sentences"])
    # data = get_first_n_sentences(data, n_sentences=10)
    # experiment("stemm_vs_lemm", data, _experiment_lemmatize_vs_stemming, ["n_topics", "preprocessing"])
    # experiment("ngrams", data, _experiment_ngrams, ["n_topics", "ngrams"])
    # experiment("filter_extremes", data, _experiment_filtering_extremes, ["n_topics", "above", "below"])

    # Model
    # experiment("n_passes", data, _experiment_n_passes, ["n_topics", "n_passes"])
    # experiment_n_topics(data)

    # Base Model
    dictionary, corpus, texts = preprocess_texts(data, {})
    model = build_lda_model(corpus, dictionary, n_topics=10, use_multicore=False)
    perplexity, coherence = evaluate_lda(model, dictionary, corpus, texts)

