import random
import string, pickle
from functools import partial
from multiprocessing import Pool

import pandas as pd
import nltk
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

from pprint import pprint

import gensim
import spacy
from gensim import corpora

from Assignments.report2.ldamallet import LdaMallet
from preprocess_nlp import preprocess_nlp

import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus

N_WORKERS = 5
RANDOM_STATE = 42


class OwnPhraser(gensim.models.phrases.Phraser):
    """ Own Phraser class to map __getitem__ to __call__ so this can be used as a function."""

    def __call__(self, sentence, *args, **kwargs):
        return self.__getitem__(sentence)


def evaluate_lda(model, dictionary, corpus, texts, calculate_coherence=True, use_multicore=False):
    """ evaluates lda model using perplexity and coherence (if calculate_coherence is True, takes a long time)"""
    # perplexity = model.log_perplexity(corpus)
    coherence_lda = None
    if calculate_coherence:
        coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                                             coherence='c_v', processes=N_WORKERS if use_multicore else 1)
        coherence_lda = coherence_model_lda.get_coherence()
    return 0, coherence_lda


def save_dictionary_corpus_texts(name, dictionary, corpus, texts):
    """ Save the dictionary, corpus and texts as dumps"""
    dictionary.save(f'dictionary_{name}.gensim')
    pickle.dump(corpus, open(f'corpus_{name}.pkl', 'wb'))
    pickle.dump(texts, open(f'texts_{name}.pkl', 'wb'))


def load_dictionary_corpus_texts(name):
    """ load dictionary, corpus and text from current directory"""
    dictionary = corpora.Dictionary.load(f'dictionary_{name}.gensim')
    corpus = pickle.load(open(f'corpus_{name}.pkl', 'rb'))
    texts = pickle.load(open(f'texts_{name}.pkl', 'rb'))
    return dictionary, corpus, texts


def process_twograms(data):
    """ """
    bigram = gensim.models.Phrases(data)
    bigram_mod = OwnPhraser(bigram)
    # with Pool(processes=N_WORKERS) as pool:
    #     data_bigram = pool.map(bigram_mod, data)
    # return data_bigram
    return [bigram_mod[d] for d in data]


def process_trigrams(data):
    """ """
    bigram = gensim.models.Phrases(data)
    trigram = gensim.models.Phrases(bigram[data])
    trigram_mod = OwnPhraser(trigram)
    # with Pool(processes=N_WORKERS) as pool:
    #     data_trigram = pool.map(trigram_mod, data)
    # return data_trigram
    return [trigram_mod[d] for d in data]


def get_first_n_sentences(data, n_sentences=10, return_text=False):
    if n_sentences > 0:
        sent_data = list()
        for text in data:
            sentences = list(nltk.sent_tokenize(text))[:n_sentences]
            if return_text:
                sent_data.append("\n".join(sentences))
            else:
                sent_data.append(sentences)
        return sent_data
    return data


def preprocess_texts(data, stages, no_below=10, no_above=0.75, ngram=1, multicore=False, dictionary=None):
    """" """

    if dictionary is None:
        if multicore:
            with Pool(processes=N_WORKERS) as pool:
                preprocessed_data = pool.map(partial(preprocess_nlp, stages=stages), data)
        else:
            preprocessed_data = [preprocess_nlp(d, stages=stages) for d in data]

        if ngram == 2:
            preprocessed_data = process_twograms(preprocessed_data)
        if ngram == 3:
            preprocessed_data = process_trigrams(preprocessed_data)

        dictionary = corpora.Dictionary(preprocessed_data)
    else:
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        preprocessed_data = data

    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in preprocessed_data]

    return dictionary, corpus, preprocessed_data


def build_lda_model(corpus, dictionary, n_topics, passes=10, alpha="auto", eta=None, use_multicore=False, ):
    try:
        if use_multicore:
            return gensim.models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=n_topics,
                                                           random_state=RANDOM_STATE, passes=passes,
                                                           per_word_topics=True,
                                                           eval_every=100, workers=N_WORKERS)
        return gensim.models.ldamulticore.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics,
                                                   random_state=RANDOM_STATE, passes=passes, per_word_topics=True,
                                                   eval_every=100, alpha=alpha, eta=eta)
    except Exception as e:
        print("LDA Exception:", e, "Len corpus:", len(corpus), "Len dictionary:", len(dictionary.keys()))
        return None


def build_mallet_model(corpus, dictionary, n_topics):
    """ Download MALLET in this folder """
    return LdaMallet("mallet-2.0.8/bin/mallet", corpus=corpus, num_topics=n_topics, id2word=dictionary, workers=5,
                     iterations=500)


def preprocess_spacy(data, n_sentences=10, trigram=False):
    spc = spacy.load('en_core_web_sm')
    data = get_first_n_sentences(data, n_sentences=n_sentences, return_text=True)

    KEEP_POS = ["NOUN", "VERB", "PROPN", "ADJ", "ADV"]  # NOUN, VERB, ADJ, ADV, PROPN
    pipe = spc.pipe(data, disable=["parser", "ner"], n_process=6)

    processed = list()
    for doc in pipe:
        p_d = list()
        for token in doc:
            if token.pos_ in KEEP_POS and not token.is_stop:
                p_d.append(token.lemma_)
        processed.append(p_d)

    if trigram:
        processed = process_trigrams(processed)

    dictionary = Dictionary(processed)
    dictionary.filter_extremes(no_below=10, no_above=0.6)
    corpus = [dictionary.doc2bow(text) for text in processed]
    return dictionary, corpus, processed


def format_topics_sentences(ldamodel, corpus, texts, n_topics=20, n_docs=100):
    """ https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#19findthemostrepresentativedocumentforeachtopic
     and some modifications TODO rewrite """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel.get_document_topics(corpus)):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 5), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts, name="Contents")
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df["Dominant_Topic"] = sent_topics_df["Dominant_Topic"].astype(int)
    a = dict()
    for i in range(n_topics):
        t = sent_topics_df[(sent_topics_df["Dominant_Topic"] == i)].sort_values(["Perc_Contribution"])
        a[i] = list(t.index)[:n_docs]
    return pd.DataFrame(a)

def convertldaGenToldaMallet(mallet_model):
    """ from https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know"""
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim



if __name__ == '__main__':

    spc = spacy.load('en_core_web_sm')

    dataset = "news_dataset_small"
    n_sentences = 50
    n_topics = 20

    reading = False

    if not reading:
        data = pd.read_csv(f"{dataset}.csv")["content"]
        data.dropna(inplace=True)

        data = get_first_n_sentences(data, n_sentences=n_sentences, return_text=True)

        KEEP_POS = ["NOUN", "VERB", "PROPN"]  # NOUN, VERB, ADJ, ADV, PROPN
        pipe = spc.pipe(data, disable=["parser", "ner"], n_process=6)

        processed = list()
        for doc in pipe:
            p_d = list()
            for token in doc:
                if token.pos_ in KEEP_POS and not token.is_stop:
                    p_d.append(token.lemma_)
            processed.append(p_d)

        # processed = process_trigrams(processed)

        dictionary = Dictionary(processed)
        dictionary.filter_extremes(no_below=10, no_above=0.6)
        corpus = [dictionary.doc2bow(text) for text in processed]
        # save_dictionary_corpus_texts(dataset + "_" + str(n_sentences), dictionary, corpus, processed)
    else:
        dictionary, _, processed = load_dictionary_corpus_texts(dataset + "_" + str(n_sentences))
        # dictionary.filter_extremes(no_below=20, no_above=0.6)
        corpus = [dictionary.doc2bow(text) for text in processed]

    model = build_lda_model(corpus, dictionary, n_topics=n_topics, use_multicore=True)
    _, coherence = evaluate_lda(model, dictionary, corpus, processed, use_multicore=True)
    print("coherence lda", coherence)

    mallet_model = build_mallet_model(corpus, dictionary, n_topics=n_topics)
    _, coherence_mallet = evaluate_lda(mallet_model, dictionary, corpus, processed, use_multicore=True)
    converted_model = convertldaGenToldaMallet(mallet_model)
    _, coherence_converted = evaluate_lda(converted_model, dictionary, corpus, processed, use_multicore=True)
    print("coherence mallet", coherence_mallet)
    print("coherence mallet converted", coherence_converted)

    # df = format_topics_sentences(convertldaGenToldaMallet(mallet_model), corpus, processed, n_docs=100, n_topics=n_topics)
    # print(df)