import os
import pickle
import zipfile
import glob
import string
import re
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Comment

import spacy

from keras.layers import Layer
from keras import backend as K


from sklearn.externals import six
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError

#designed to encode a text to the averaged of its embeddings. Could also be used to return indexes of a texts  words
#not certain
class TextEncoder_old(object):
    def __init__(self, embedding_dict, embedding_dims=None):
        self.embedding_dict=embedding_dict
        #simple pattern for tokens
        #at least trow "word" characters
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        
        #if the embedding dims are not supplied, infer from "first" item in dict
        if embedding_dims is None:
            self.embedding_dims = self.embedding_dict[list(self.embedding_dict.keys())[0]].shape[0]
        else:
            self.embedding_dims = embedding_dims

        self.word_index = {key:i for i, key in enumerate(self.embedding_dict)}
    def tokenize(self, text):
        if not isinstance(text, (bytes, str)):
            raise ValueError("text must be a string object!")
        return self.token_pattern.findall(text.lower())
    
    def embed(self, text):
        tokens = self.tokenize(text)
        mean=np.zeros(self.embedding_dims, dtype=np.float32)
        sum_wght=0
        for token in tokens:
            try:
                vector = self.embedding_dict[token]
                mean, sum_wght = self.__accum_mean(vector, mean, sum_wght)
            except KeyError:
                continue        
        return mean
    def __accum_mean(self, vec, prev_mean, prev_sum_wght):
        new_sum_weight = prev_sum_wght+1
        return (vec + prev_mean * (prev_sum_wght/new_sum_weight))/new_sum_weight, new_sum_weight

#should make a class that wraps the vecotrizer...
class TextEncoder(object):
    def __init__(self, vectorizer, max_features=2000):
        self.vectorizer = vectorizer
        self.dtm = None
        self.max_features = max_features
        
    def __len__(self):
        if self.dtm is not None:
            return self.dtm.shape[0]
        else:
            raise NotFittedError("fit this instance first")
    
    def fit(self, docs, max_features=2000):
        #if single string, make intpo tuple
        if isinstance(docs, six.string_types):
            docs = (docs,)
        self.dtm = self.vectorizer.transform(docs)
        return self.dtm
    
    def __getitem__(self, key):
        wghts = np.zeros((self.max_features), dtype = np.float32)
        feature_indices = np.zeros((self.max_features), dtype = np.int32)
        
        indices = np.random.permutation(np.arange(self.dtm.indptr[key],self.dtm.indptr[key+1]))
        
        last_feature = min(self.max_features, len(indices))
        indices = indices[:last_feature]
        feature_indices[:last_feature] = self.dtm.indices[indices]
        wghts[:last_feature] = self.dtm.data[indices]
        
        return feature_indices, wghts

#USED TO CHECK IF THE TEXT IN HTML DOC IS VISIBLE
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

##USED TO JOIN TEXT WITHIN PARAGRAP
def concatenate_texts(texts):
    return u"".join(t for t in texts)

#GETS TEXT FROM HTML
#JOIN PARAGRAPHS TOGETHER WITH 2 NEWLINES (BREAKS)
def text_from_html(html):    
    soup = BeautifulSoup(html, "html.parser")
    #first get paragraphs
    paragraphs = soup.findAll("p")
    #to exclude futenbeg headers
    paragraphs = [paragraph.findAll(text=True) for paragraph in paragraphs]
    paragraphs = [filter(tag_visible, paragraph) for paragraph in paragraphs]
    paragraphs = [concatenate_texts(paragraph) for paragraph in paragraphs]
    return u"\n\n".join(paragraph.strip() for paragraph in paragraphs)
#Gets all html docs in a directory
def get_chapters(directory, min_length=0):
    chapters=[]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".html"):
            file_path=os.path.join(directory, filename)
            chapter=text_from_html(file_path)
            #filter out ones that are too short
            if len(chapter) >= min_length:
                chapters.append(chapter)
            continue
        else:
            continue
    return chapters
#MEANT FOR EPUBS
#EXTRACTS ALL HTML, HTM OR XHTML FILES
#TREATS EACH SEPERATE FILE AS A SEPERATE CHAPER
def chapters_from_archive(file, min_length=1000):
    archive = zipfile.ZipFile(file, 'r')
    files = [file for file in archive.namelist() if file.endswith((".htm","html","xhtml"))]
    chapters=[]
    for file in files:
        html=archive.read(file)
        chapter=text_from_html(html)
        #filter out ones that are too short
        if len(chapter) >= min_length:
            chapters.append(chapter)
            continue
        else:
            continue
    return chapters

class WghtdAverage(Layer):
    def __init__(self, **kwargs):
        super(WghtdAverage, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, inputs, mask = None):
        #num willl have dimensions: dimension batch_size X embedding_dims X 1
        num = inputs[0] * inputs[1]
        return K.sum(num, axis=1) / (K.sum(inputs[1], axis=1) + K.epsilon())

def sparse_softmax_cross_entropy_with_logits (y_true, y_pred):
    return K.sparse_categorical_crossentropy(target=y_true, output=y_pred, from_logits=True)


class LemmaTokenizer(object):
    def __init__(self):
        self.nlp = spacy.load("en")
    def __only_letters(self, string):
        return re.match("^[a-z]+$", string) != None
    def __call__(self, doc):
         return [token.lemma_ for token in self.nlp(doc) if self.__only_letters(token.lemma_)]


def only_letters(string):
    return re.match("^[a-z]+$", string) != None

nlp = spacy.load("en")
def lemma_tokenize(doc):
    return [token.lemma_ for token in nlp(doc) if only_letters(token.lemma_)]


class TopicModeler(object):
    def __init__(self, n_topics=32):
        self.vectorizer = TfidfVectorizer(tokenizer=lemma_tokenize, norm="l1", strip_accents="ascii",
                             lowercase=False, use_idf=True, stop_words="english", min_df=5, max_df=0.9)
        self.model = NMF(n_topics)
    def fit_transform(self, texts):
        dtm = self.vectorizer.fit_transform(texts)
        return self.model.fit_transform(dtm)

    def transform(self, texts):
        dtm = self.vectorizer.transform(texts)
        return self.model.transform(dtm)
    def top_words(self, n_top_words = 20):
        feature_names = self.vectorizer.get_feature_names()
        top_words = []

        for topic_idx, topic in enumerate(self.model.components_):
            top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        return top_words