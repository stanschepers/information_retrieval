Stan Schepers (*20163280*) — August 2021 — Information Retrieval

# Report 1: Lucene

## Description Lucene

In this section we will discuss the functionality of Lucene. Most of the information below is based on the documentation of Lucene. [^Documentation]

[^Documentation]: https://lucene.apache.org/core/8_9_0/core/index.html

### Document Analysis and Indexing

Lucene has several different analyses for indexing content in different ways and languages. Some examples that were seen in class are **MinHash** and **Shingle**. It is also possible to create your own `Analyzer` and `Stemmer` to analyse documents.

Indexing in Lucene is handled by an `IndexWriter` that stores terms and document ids in an **inverted index** using indexable **Skip-Lists** instead of B-Trees.[^SkipLinks] The index will be predominantly be stored in a single file.

[^SkipLinks]: https://stackoverflow.com/questions/2602253/how-does-lucene-index-documents

### Query Processing

Different types of queries are possible e.g.:

- **Wildcard query**: 
- **Multi term query**: allows a search documentents containing a subset of given terms.
- **Phrase query**: allows a search documentents containing multiple given terms in a certain order.
- **Fuzzy query**: allows a search for non exact words (up to 2 edits) using **Levenshtein** algorithm. This could be used as spell correction. There are also other ways to implement spell corrections in Lucene.
- **Regex query**: allows a search using RegEx.

### Document Search and Retrieval

Lucene provides different ranking methods that were seen in class: **vector-space models**, **theory-based probabilistic model** (e.g. *BM-25*) and language-based models.

The vector-space model of Lucene is also discussed in class. 



### Lucene vs. Solr

Solr is a web-based[^Solr] 

[^Solr]: http://www.lucenetutorial.com/lucene-vs-solr.html







