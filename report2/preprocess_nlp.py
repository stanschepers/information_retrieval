"""
Based on https://github.com/nikhiljsk/preprocess_nlp
"""

import re

import nltk
import string
import unicodedata
import contractions


def remove_tags(text):
    """
    Helper function for preprocess_nlp which is used to remove HTML Tags, Accented chars, Non-Ascii Characters, Emails and URLs
    
    :param text: A string to be cleaned
    
    <Returns the cleaned text>
    """
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8',
                                                                                     'ignore')  # Remove Accented characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove Non-Ascii characters
    text = re.sub("[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", '', text)  # Remove Emails
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    return text


def preprocess_nlp(text, stages=None, include_n_sentences=None):
    """
    Function to preprocess a list of strings using standard nlp preprocessing techniques. 
    Note: NaN values are not supported and would raise a runtime Exception (Only <str> type is supported)
    
    :param strings: A list of strings to be preprocessed
    :param stages: A dictionary with keys as stages and values as Boolean/Integer. Can be used to customize the stages in preprocessing
    :param ind: Automatically called while using 'async_call_preprocess', indicates Index of process call
    :param return_dict: Automatically called while using 'async_call_preprocess', stores the preprocessed content for each process call
    
    (Default parameters for stages):
    {'remove_tags_nonascii': True, 
    'lower_case': True,
    'expand_contractions': False, 
    'remove_punctuation': True, 
    'remove_escape_chars': True, 
    'remove_stopwords': False, 
    'remove_numbers': True, 
    'lemmatize': False, 
    'stemming': False, 
    'min_word_len': 2}
    
    :return preprocessed strings
    """

    default_stages = {'remove_tags_nonascii': True,
                      'lower_case': True,
                      'expand_contractions': False,
                      'remove_escape_chars': True,
                      'remove_punctuation': True,
                      'remove_stopwords': True,
                      'remove_numbers': True,
                      'lemmatize': False,
                      'stemming': False,
                      'min_word_len': 2  # e.g. U.S.
                      }

    # Update the key-values based on dictionary passed

    if stages is not None:
        default_stages.update(stages)

    # Initializations
    cached_stopwords = nltk.corpus.stopwords.words('english')
    cached_stopwords.extend(["said"])
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Iterate over sentences
    if default_stages['remove_tags_nonascii']:  # Accented Chars, Emails, URLs, Non-Ascii
        text = remove_tags(text)

    if default_stages['lower_case']:  # Lower-case the sentence
        text = text.lower().strip()

    if default_stages['expand_contractions']:  # Expand contractions
        text = contractions.fix(text)

    if default_stages['remove_escape_chars']:  # Remove multiple spaces & \n etc.
        text = re.sub('\s+', ' ', text)

    if default_stages['remove_punctuation']:  # Remove all punctuations
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(
            string.punctuation)))  # If replace with punct with space
        # sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # If replace without space

    if default_stages['remove_numbers']:  # Remove digits
        text = text.translate(str.maketrans('', '', string.digits))

    tokenized_text = nltk.word_tokenize(text)

    if default_stages['remove_stopwords']:  # Remove Stopwords
        tokenized_text = [word for word in tokenized_text if word not in cached_stopwords]

    if default_stages['lemmatize']:  # Lemmatize words
        tokenized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]

    if default_stages['stemming']:  # Stemming words
        tokenized_text = [stemmer.stem(word) for word in tokenized_text]

    return tokenized_text
