# ====================================================================
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ====================================================================
import re
import sys, lucene, unittest
import os, shutil

import emoji

from java.io import StringReader
from java.lang import System
from java.nio.file import Path, Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import \
    Document, Field, StoredField, StringField, TextField
from org.apache.lucene.index import \
    IndexOptions, IndexWriter, IndexWriterConfig, DirectoryReader, \
    FieldInfos, MultiFields, MultiTerms, Term
from org.apache.lucene.util import PrintStreamInfoStream
from org.apache.lucene.queryparser.classic import \
    MultiFieldQueryParser, QueryParser
from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.pylucene.search.similarities import PythonClassicSimilarity
from org.apache.pylucene.analysis import PythonAnalyzer, PythonTokenFilter
from org.apache.lucene.search.similarities import BM25Similarity, BooleanSimilarity
from org.apache.lucene.analysis.core import WhitespaceTokenizer, LowerCaseFilter, StopFilter
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute


class PyLuceneIndex:
    def __init__(self, store_directory="index", analyser=None, similarity=None, default_k=10):
        self.store_directory = store_directory
        self.analyser = analyser if analyser is not None else StandardAnalyzer()
        self.similarity = similarity if similarity is not None else PythonClassicSimilarity()
        self.store = self.new_store()
        self._writer = None
        self._reader = None
        self.default_k = default_k
        self.pause_closing = False
        self._start()

    def _start(self):
        writer = IndexWriter(self.store, IndexWriterConfig(self.analyser))
        writer.close()

    @property
    def writer(self):
        return self._writer if self._writer else self.new_writer()

    @writer.setter
    def writer(self, writer):
        self._writer = writer

    @property
    def reader(self):
        return self._reader if self._reader else self.new_reader()

    @reader.setter
    def reader(self, reader):
        self._reader = reader

    def __setitem__(self, key, value):
        writer = self.writer
        document = Document()
        document.add(Field("key", key, StringField.TYPE_STORED))
        document.add(Field("value", value, TextField.TYPE_STORED))
        writer.addDocument(document)
        self.close(writer)

    def __getitem__(self, item):
        if type(item) is tuple:
            query, k = item
        else:
            query, k = item, self.default_k
        if type(query) is str:
            query = QueryParser("value", self.analyser).parse(query)
        reader = self.reader
        searcher = self.new_searcher(reader)
        results = [searcher.doc(s.doc).getField("key").stringValue() for s in searcher.search(query, k).scoreDocs]
        self.close(reader)
        return results

    def __delitem__(self, key):
        writer = self.writer
        writer.deleteDocuments(Term("key", key))
        self.close(writer)

    def close(self, *args):
        if not self.pause_closing:
            for arg in args:
                if arg is not None:
                    arg.close()

    def new_store(self):
        return SimpleFSDirectory(Paths.get(self.store_directory))

    def new_searcher(self, reader):
        searcher = IndexSearcher(reader)
        searcher.setSimilarity(self.similarity)
        return searcher

    def new_writer(self, create=False):
        config = IndexWriterConfig(self.analyser)
        config.setSimilarity(self.similarity)
        if create:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        return IndexWriter(self.new_store(), config)

    def new_reader(self):
        return DirectoryReader.open(self.store)

    def close_index(self):
        self.store.close()


class PyLuceneMMapIndex(PyLuceneIndex):
    """ PyLucene Index using MMapDirectory """

    def new_store(self):
        return MMapDirectory(Paths.get(self.store_directory))


class PyLuceneIndexSession:
    """ Session object """

    def __init__(self, index: PyLuceneIndex):
        self.index = index

    def __enter__(self):
        self.index.pause_closing = True
        self.index.writer = self.index.new_writer()
        self.index.reader = self.index.new_reader()
        return self.index

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.index.pause_closing = False
        self.index.close(self.index.writer, self.index.reader)
        self.index.writer, self.index.reader = None, None


class EmojiFilter(PythonTokenFilter):

    def __init__(self, in_stream):
        PythonTokenFilter.__init__(self, in_stream)
        self.term = self.addAttribute(CharTermAttribute.class_)
        # Get tokens.
        tokens = []
        while in_stream.incrementToken():
            tokens.append(emoji.demojize(self.term.term(), delimiters=("", "")))
        # Setup iterator.
        self.iter = iter(tokens)

    def incrementToken(self):
        try:
            self.term.setTermBuffer(next(self.iter))
        except StopIteration:
            return False
        return True


class StAnalyser(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        source = WhitespaceTokenizer()
        # result = EmojiFilter(source)
        result = LowerCaseFilter(source)
        # result = StopFilter(result, ["stan"])
        return Analyzer.TokenStreamComponents(source, result)


def test_query(query, should="", length=False):
    results = index[query]
    if length:
        print("query:", query, "n results:", len(results), "expected to be", should)
    else:
        print("query:", query, "results:", results, "expected to be", should)


if __name__ == "__main__":
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    import csv

    """ Creating own analyser """

    """Creating index using non-default similarity and own analyser"""
    # index_path = "index"
    # if os.path.exists(index_path):
    #     shutil.rmtree(index_path)

    index = PyLuceneIndex("index")

    try:
        """ Indexing all data """
        with open('data/people_wiki.csv', 'r') as read_obj:
            data = list(map(tuple, csv.reader(read_obj)))
        with PyLuceneIndexSession(index):
            for i, (_, name, text) in enumerate(data[1:]):
                index[name] = text
                if i % 1000 == 0:
                    print(i, "/", len(data) - 1)

        """ Testing normal quering """

        test_query("born", "10 as k=10", length=True)
        test_query(("born", 1), "1 as k=1", length=True)

        index_path2 = "index2"
        if os.path.exists(index_path2):
            shutil.rmtree(index_path2)

        index = PyLuceneIndex(index_path2)

        # """ Indexing grouth truth"""
        with PyLuceneIndexSession(index):
            index["Stan Schepers"] = "Computer Science Master Student at UAntwerp, born in 1998.."
            index[
                "Donald Knuth"] = "computer scientist and professor born in 1938. Has won the A.M. Turing Award. He opened a building at Campus Middelheim at UAntwerp."
            index[
                "Ada Lovelace"] = "was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation, and to have published the first algorithm intended to be carried out by such a machine. As a result, she is often regarded as the first computer programmer."
            index["N.N."] = "foobarbarbarfoo"

        """ Deleting data out of index"""

        test_query("foobarbarbarfoo", "N.N.")
        del index["N.N."]
        test_query("foobarbarbarfoo", "empty")

        """ Changing similarity """
        index.similarity = BM25Similarity()

        """ Boolean Queries """
        test_query("computer AND professor", "Knuth")
        test_query("(student or professor) AND UAntwerp", "Knuth and Stan")
        test_query("NOT professor", "Ada and Stan")

        """ Fuzzy queries"""
        test_query("computer AND proffesor~", "Knuth as he is professor")
        test_query("computer AND professor", "Knuth as he is professor")

        """ Wildcard queries """
        test_query("pro*e*or", "Knuth as he is pro(f)e(ss)or")  # how to you write proffesor or professor

        """ Proximity Search """
        test_query('"computer UAntwerp"~6', "only Stan")
        test_query('"computer UAntwerp"~20', "Stan and Knuth")

        emoji_path = "emoji_index"
        if os.path.exists(emoji_path):
            shutil.rmtree(emoji_path)

        emoji_index = PyLuceneIndex(emoji_path)
        stanalyser = StAnalyser()
        emoji_index = PyLuceneIndex(analyser=stanalyser)

        emoji_index["happy person"] = "Hi, I am a very happy person! "
        emoji_index["Neil Amstrong"] = "I am an astronaut and I flew a rocket!"
        emoji_index["Stan Schepers"] = "Hi, I am a cs student."

        print(emoji_index["🙂"])  # output: ["happy person"]
        # print(emoji_index["champagne"])  # output ["happy person"]
        print(emoji_index["🚀"])  # output: ["Neil Amstrong"]
        print(emoji_index["👨‍🎓"])  # output: ["Stan Schepers"]


    finally:
        # clean up for Github
        index.close_index()
        index_path = "index"
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        index_path2 = "index2"
        if os.path.exists(index_path2):
            shutil.rmtree(index_path2)
        emoji_path = "emoji_index"
        if os.path.exists(emoji_path):
            shutil.rmtree(emoji_path)
