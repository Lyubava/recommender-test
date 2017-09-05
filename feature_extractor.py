#!/usr/bin/python3
# -*-coding: utf-8 -*-
"""
Make sure to run data_reader.py before proceed
"""

import json
import pickle
from nltk import ngrams
import os
import numpy
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from zope.interface import implementer, Interface

con = psycopg2.connect(
    user="postgres", host="localhost", dbname="olx_data", password="postgres")
con.autocommit = True
cur = con.cursor()


class Preprocessor(object):
    """
    Basic text preprocessing
    """
    def __init__(self):
        self.stop_words = \
            ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
             'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
             'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
             'into', 'through', 'during', 'before', 'after', 'above', 'below',
             'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
             'under', 'again', 'further', 'then', 'once', 'here', 'there',
             'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
             'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
             't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
             'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
             'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
             'condition', 'great', 'good', 'selling', 'sale', 'service',
             'urgent', 'price', 'viewing', 'quality']

    def preprocessed(self, str_input):
        """
        Preprocess input string: removes punctuation and other symbols except
        letters, lower case, stop words filter
        :param str_input: input string
        :return: output string
        """
        for char in str_input:
            if not char.isalpha():
                str_input = str_input.replace(char, " ")
        words = str_input.strip().lower().split(" ")
        filtered_words = [
            word for word in words if (word not in self.stop_words
                                       and len(word) > 2)]
        return " ".join(filtered_words)


class IFeatureExtractor(Interface):
    def get_vector_feature(text_feature, *args):
        """
        Calculates vector feature from text feature
        """


class FeatureExtractorBase(Preprocessor):
    """
    Base class for feature extraction
    """
    def __init__(self):
        super(FeatureExtractorBase, self).__init__()
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.n_gram_lens = (1, 2, 3, 4)
        self.threshold_max_count = 2
        self.threshold_count = 0.5
        self.min_feature_len = 3

    def find_most_frequently_n_grams(self, data_set):
        """
        Finds most frequently n-grams in all dataset
        :param data_set: type of data set (train or test)
        :return: most frequently n-grams by n and categories
        """
        query = "SELECT item_id, listing_title, category_l3_name_en FROM " \
                "samples_%s;" % data_set
        cur.execute(query)
        res = cur.fetchall()
        print("Will find most frequently n grams...")
        text_data_by_categories = {}
        for row in res:
            item_id, listing_title, category_l3_name_en = row
            if category_l3_name_en not in text_data_by_categories:
                text_data_by_categories[category_l3_name_en] = [
                    self.preprocessed(listing_title)]
            else:
                text_data_by_categories[category_l3_name_en].extend(
                    [self.preprocessed(listing_title)])
        n_grams_by_count = {}
        for n_gram_len in self.n_gram_lens:
            n_grams_by_categories = {}
            for category, text_data in text_data_by_categories.items():
                n_grams_by_frequency = {}
                for sentence in text_data:
                    n_grams = ngrams(sentence.split(), n_gram_len)
                    for n_gram in n_grams:
                        if n_gram not in n_grams_by_frequency:
                            n_grams_by_frequency[n_gram] = 1
                        else:
                            n_grams_by_frequency[n_gram] += 1
                n_grams_by_categories[category] = n_grams_by_frequency
            n_grams_by_count[n_gram_len] = n_grams_by_categories
        most_frequently_n_grams_by_count = {}
        for n_gram_len, values in n_grams_by_count.items():
            most_frequently_n_grams_by_cat = {}
            for category, n_grams_by_frequency in values.items():
                most_frequently_n_grams_by_cat[category] = []
                counts_list = list(
                    n_grams_by_count[n_gram_len][category].values())
                if counts_list:
                    max_count = max(counts_list)
                else:
                    max_count = 0
                for n_gram, count in n_grams_by_frequency.items():
                    if (max_count > self.threshold_max_count and
                            count >= self.threshold_count * max_count):
                        most_frequently_n_grams_by_cat[category].append(
                            list(set(n_gram)))
            most_frequently_n_grams_by_count[n_gram_len] = \
                most_frequently_n_grams_by_cat
        return most_frequently_n_grams_by_count

    def extract_features_per_item(self, res, fr_n_grams):
        """
        Extracts text features from item info given most frequently n-grams
        :param res:  item info
        :param fr_n_grams: most frequently n-grams
        :return: set of text features
        """
        features = set()
        listing_title, listing_description, \
            listing_price, category_sk, category_l1_name_en, \
            category_l2_name_en, category_l3_name_en, listing_latitude, \
            listing_longitude = res
        preprocessed_title = self.preprocessed(listing_title).split(" ")
        for n_gram_len in self.n_gram_lens:
            n_grams = ngrams(preprocessed_title, n_gram_len)
            for n_gram in n_grams:
                n_gram_set = set(n_gram)
                n_gams_by_category = fr_n_grams[str(n_gram_len)]
                if category_l3_name_en not in n_gams_by_category:
                    continue
                for elem in n_gams_by_category[
                        category_l3_name_en]:
                    if set(elem) == n_gram_set:
                        features = features.union(n_gram_set)
        if len(features) < self.min_feature_len:
            features = features.union(set(preprocessed_title))
        if len(features) < self.min_feature_len:
            features = features.union(
                set(self.preprocessed(listing_description).split(" ")))
        return features

    def extract_features_per_item_db(self, item_id, fr_n_grams, data_set):
        """
        Extracts text features from item id given most frequently n-grams
        :param item_id: input item id
        :param fr_n_grams: most frequently n-grams
        :param data_set: type of data set (train or test)
        :return: set of text features
        """
        query = "SELECT listing_title, listing_description, listing_price, " \
                "category_sk, category_l1_name_en, category_l2_name_en, " \
                "category_l3_name_en, listing_latitude, listing_longitude " \
                "FROM samples_%s WHERE item_id=%%s;" % data_set
        cur.execute(query, (item_id,))
        res = cur.fetchone()
        if not res:
            raise AttributeError(
                "No data for item %s in the database" % item_id)
        return self.extract_features_per_item(res, fr_n_grams)

    def extract_features(self, fr_n_grams, data_set, *args):
        """
        Exracts text and vectors features for all dataset. Append text and
        vectors features per item id in PostgreSQL database
        :param fr_n_grams: most frequently n-grams
        :param data_set: type of data set (train or test)
        :param args: arguments for get_vector_feature() function which should
        be implemented in child class
        """
        print("Will extract features for %s" % data_set)
        query = "SELECT EXISTS (SELECT 1 FROM information_schema.columns " \
                "WHERE table_name ='samples_%s' AND " \
                "column_name='text_feature');" % data_set
        cur.execute(query)
        row = cur.fetchone()
        if not row[0]:
            query = "ALTER TABLE samples_%s ADD COLUMN text_feature character " \
                    "varying(500)" % data_set
            cur.execute(query)
            query = "ALTER TABLE samples_%s ADD COLUMN vector_feature double " \
                    "precision[]" % data_set
            cur.execute(query)
        query = "SELECT item_id FROM samples_%s;" % data_set
        cur.execute(query)
        res = cur.fetchall()
        item_count = 0
        for row in res:
            if not row:
                continue
            item_id, = row
            item_count += 1
            print("Item id: %s. Item count: %s" % (item_id, item_count))
            item_features = self.extract_features_per_item_db(
                item_id, fr_n_grams, data_set)
            text_feature = " ".join(item_features)
            vector_feature = self.get_vector_feature(text_feature, *args)
            query = "UPDATE samples_%s SET text_feature = %%s, vector_feature " \
                    "= %%s WHERE item_id = %%s;" % data_set
            cur.execute(query, (text_feature, vector_feature, item_id))


@implementer(IFeatureExtractor)
class FeatureExtractorMI(FeatureExtractorBase):
    """
    Category-feature mutual information based feature extractor
    """
    @staticmethod
    def get_categories(data_set):
        query = "SELECT category_l1_name_en,category_l2_name_en, " \
                "category_l3_name_en FROM samples_%s;" % data_set
        cur.execute(query)
        res = cur.fetchall()
        distinct_categories_l1_name_en = set()
        distinct_categories_l2_name_en = set()
        distinct_categories_l3_name_en = set()
        for row in res:
            category_l1_name_en, category_l2_name_en, category_l3_name_en = row
            distinct_categories_l1_name_en.add(category_l1_name_en)
            distinct_categories_l2_name_en.add(category_l2_name_en)
            distinct_categories_l3_name_en.add(category_l3_name_en)
        return distinct_categories_l1_name_en, distinct_categories_l2_name_en, \
            distinct_categories_l3_name_en

    @staticmethod
    def get_disctinct_features(
            path_to_features, path_to_disctinct_features):
        with open(path_to_features, "rb") as fin:
            features = pickle.load(fin)
        disctinct_features = set()
        for seller_id, seller_items in features.items():
            for item_id, item_features in seller_items.items():
                for feature in item_features:
                    disctinct_features.add(feature)
        disctinct_features = list(disctinct_features)
        with open(path_to_disctinct_features, "wb") as fin:
            pickle.dump(disctinct_features, fin)

    def get_matrix(
            self, path_to_features, path_to_disctinct_features, data_set):
        categories_l1, categories_l2, categories_l3 = self.get_categories(
            data_set)
        disctinct_categories = list(
            categories_l1 | categories_l2 | categories_l3)
        with open(path_to_features, "rb") as fin:
            features = pickle.load(fin)
        with open(path_to_disctinct_features, "rb") as fin:
            disctinct_features = pickle.load(fin)
        matrix = numpy.zeros(
            [len(disctinct_features), len(disctinct_categories)])
        for seller_id, seller_items in features.items():
            for item_id, item_features in seller_items.items():
                for feature in item_features:
                    query = "SELECT category_l1_name_en,category_l2_name_en, " \
                            "category_l3_name_en FROM samples_%s WHERE " \
                            "item_id='%%s';" % data_set
                    cur.execute(query, (item_id, ))
                    res = cur.fetchone()
                    category_l1_name_en, category_l2_name_en, \
                        category_l3_name_en = res
                    for cat in (category_l1_name_en, category_l2_name_en,
                                category_l3_name_en):
                        category_index = disctinct_categories.index(cat)
                        feature_index = disctinct_features.index(feature)
                        matrix[feature_index][category_index] += 1
        import pdb
        pdb.set_trace()

    @staticmethod
    def get_vector_feature(text_feature, *args):
        # TODO: implement
        pass


@implementer(IFeatureExtractor)
class FeatureExtractorW2V(FeatureExtractorBase):
    """
    Word2vec based feature extractor
    """
    def __init__(self):
        super(FeatureExtractorW2V, self).__init__()
        self.vector_length = 300
        self.path_to_ngrams = os.path.join(
            self.base_path, "most_frequently_n_grams.json")
        self.path_to_features_classes = os.path.join(
            self.base_path, "features_classes.pickle")
        self.path_to_word_tfidf = os.path.join(
            self.base_path, "word_tfidf.pickle")

    @staticmethod
    def get_vector_from_word(nlp_model, word, word2tfidf):
        """
        Calcalutes vector for given word with tfidf weights
        :param nlp_model: word2vec model
        :param word: input word
        :param word2tfidf: words with tfidf dictionary
        :return: output vector
        """
        try:
            vec = nlp_model[word]
        except:
            print(word)
            return
        if not word2tfidf or word not in word2tfidf:
            idf = 0
            print(word)
        else:
            idf = word2tfidf[word]
        return vec * idf

    @staticmethod
    def get_word2tfidf(features_classes):
        """
        Calculates words with tfidf dictionary
        :param features_classes: all text features
        :return: words with tfidf dictionary
        """
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(features_classes)
        return dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    @staticmethod
    def load_model_from_vec():
        """
        Loads word2vec model from .vec format
        :return: word2vec model
        """
        from gensim.models.keyedvectors import KeyedVectors

        basepath = os.path.dirname(os.path.abspath(__file__))
        path_to_model = os.path.join(basepath, 'wiki.en.vec')
        lan_model = KeyedVectors.load_word2vec_format(
            path_to_model, binary=False)
        return lan_model

    def get_vector(self, text, nlp_model, word2tfidf):
        """
        Calcalutes mean vector for given text
        :param text: input text
        :param nlp_model: word2vec model
        :param word2tfidf: words with tfidf dictionary
        :return: output vector
        """
        words = text.split(" ")
        mean_vecs = numpy.zeros([len(words), self.vector_length])
        index = 0
        for word in words:
            mean_vec = self.get_vector_from_word(
                nlp_model, word, word2tfidf)
            if mean_vec is not None:
                mean_vecs[index] = mean_vec
            index += 1
        mean_vec = list(mean_vecs.mean(axis=0))
        return mean_vec

    def get_all_txt_features(self, fr_n_grams, data_set):
        """
        Finds all text features in dataset
        :param fr_n_grams: most frequently n-grams
        :param data_set: type of data set (train or test)
        :return: all text features
        """
        query = "SELECT listing_title, listing_description, listing_price, " \
                "category_sk, category_l1_name_en, category_l2_name_en, " \
                "category_l3_name_en, listing_latitude, listing_longitude " \
                "FROM samples_%s;" % data_set
        cur.execute(query)
        res = cur.fetchall()
        features_classes = []
        for row in res:
            if not row:
                continue
            item_features = self.extract_features_per_item(
                row, fr_n_grams)
            item_features_text = " ".join(item_features)
            features_classes.append(item_features_text)
        return features_classes

    def get_vector_feature(self, text_feature, *args):
        nlp_model, word2tfidf = args
        return self.get_vector(text_feature, nlp_model, word2tfidf)

    def load_model_from_pickle(self):
        """
        Loads word2vec model from .pickle format (much faster than from .vec)
        :return: word2vec model
        """
        print("Will load nlp model...")
        basepath = os.path.dirname(os.path.abspath(__file__))
        path_to_model = os.path.join(basepath, "nlp_model.pickle")
        if os.access(path_to_model, os.F_OK):
            with open(path_to_model, "rb") as fin:
                nlp_model = pickle.load(fin)
        else:
            nlp_model = self.load_model_from_vec()
            with open(path_to_model, "wb") as fout:
                pickle.dump(nlp_model, fout)
        print("Loaded")
        return nlp_model

    def get_features_for_test(self):
        """
        Extracts features for test set
        """
        with open(self.path_to_ngrams, "r") as fin:
            frequently_n_grams = json.load(fin)
        with open(self.path_to_word_tfidf, "rb") as fin:
            word_tfidf = pickle.load(fin)
        nlp_model_ = self.load_model_from_pickle()
        self.extract_features(
            frequently_n_grams, "test", nlp_model_,
            word_tfidf)

    def get_features_for_train(self):
        """
        Extracts features for train set
        """
        set_of_data = "train"
        frequently_n_grams = self.find_most_frequently_n_grams(
            set_of_data)
        with open(self.path_to_ngrams, "w") as fout:
            json.dump(frequently_n_grams, fout)
        with open(self.path_to_ngrams, "r") as fin:
            frequently_n_grams = json.load(fin)
        feat_classes = self.get_all_txt_features(
            frequently_n_grams, set_of_data)
        word_tfidf = self.get_word2tfidf(feat_classes)
        with open(self.path_to_word_tfidf, "wb") as fout:
            pickle.dump(word_tfidf, fout)
        nlp_model_ = self.load_model_from_pickle()
        self.extract_features(
            frequently_n_grams, set_of_data, nlp_model_,
            word_tfidf)

if __name__ == "__main__":
    feature_extractor = FeatureExtractorW2V()
    feature_extractor.get_features_for_train()
    feature_extractor.get_features_for_test()