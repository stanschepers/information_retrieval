Stan Schepers — August 2021 — Information Retrieval

# Report 2: Topic Modeling

## Introduction

In this report I will discuss with topic modeling of news articles. I will conduct several different experiments in the different stages of the process: text preprocessing, choosing the right model, deciding the hyperparameters, etc.

[TOC]

## Experiments

### Data, Preprocessing and Model

For these experiments I will be using the given dataset containing 141,585 news articles. The dataset contains for each article an ID, the title, the news outlet, the author, publication date and the content of the article itself.

Preprocessing is in my/everyone's opinion 80% of the actual work. Choosing and training the model is actually secondary task. So I will be focussing a lot on the preprocessing steps in these experiments.

The default preprocessing steps that were taken are **tokenisation**, **stopword removal**, **case-folding**. Also accents, non-ASCII characters, escape characters, e-mails and URLs were removed. When not specified also **lemmazation** is used. The minimum length of a word has to be two. Unless specified only the first 50 sentences are used to reduce the running time significant. In the experiment 'Number of Sentences' we will learn that this also will increase the performance of the models.

For the topic modeling I will be using the Latent Direchlet Allocation (LDA) Model from the library `gensim`[^gensim]. In the experiments around the preprocessing phase we will be using the default hyperparameters and the number of topic 10.

[^gensim]: https://radimrehurek.com/gensim/auto_examples/index.html#documentation

### Workflow and Evaluation

All comparing experiments are performed 50 times each using a

Performance is measured using the **coherence score**. Cohernce score 

**Performance Base Model**

### Preprocessing Experiments

**Stemming vs. Lemmazation**

| n_topics | preprocessing | coherence_mean |
| -------: | :------------ | -------------: |
|       10 | lemmatization |       0.516882 |
|       10 | stemming      |         0.5357 |
|       50 | lemmatization |       0.528316 |
|       50 | stemming      |       0.513829 |

**Number of Sentences**

- Min/mean/max sentences in an article: 

| n_topics | n_sentences | coherence_mean |
| -------: | ----------: | -------------: |
|       10 |         all |       0.447265 |
|       10 |          10 |   **0.537237** |
|       10 |          50 |       0.473598 |
|       10 |         100 |       0.453009 |

**Filtering Extremes**

**N-grams**

| n_topics | ngrams | coherence_mean |
| -------: | -----: | -------------: |
|       10 |      1 |       0.537237 |
|       10 |      2 |       0.515505 |
|       10 |      3 |       0.517331 |
|       50 |      1 |       0.542408 |
|       50 |      2 |       0.536831 |
|       50 |      3 |       0.519965 |

### Preprocessing using SpaCy

### Model Experiments

**Number of Topics**

**How to choose an optimal $k$​​​ value?**

**Which Model To Use?**

### 



## Final Model and Extra Insights

### Choosen Preprocessing

### Choosen Model and Hyperparameters

### Results Top 100 with $k=20$​

### Visualization

