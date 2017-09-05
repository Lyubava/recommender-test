#!/usr/bin/python3
# -*-coding: utf-8 -*-
"""
Make sure to run feature_extractor.py and model_clusters.py
(if result_clusters.pickle is not exists) before proceed
"""

from heapq import heappush, nsmallest
import os
import pickle
import numpy
import psycopg2

con = psycopg2.connect(
    user="postgres", host="localhost", dbname="olx_data", password="postgres")
con.autocommit = True
cur = con.cursor()


class Recommender(object):
    """
    Predicts similar items
    """
    def __init__(self, to_result_clusters, top_clusters, top_recommendations):
        with open(to_result_clusters, "rb") as fin:
            self.merged_clusters = pickle.load(fin)
            _ = pickle.load(fin)
            self.centroids = pickle.load(fin)
        self.top_clusters = top_clusters
        self.top_recommendations = top_recommendations

    @staticmethod
    def predict_cluster(feature, centroids, n_closest):
        """
        Find n closest clusters given item feature
        :param feature: input feature
        :param centroids: centroids of clusters model
        :param n_closest: top n_closest
        :return: n_closest clusters indicies
        """
        clusters_heap = []
        for cluster_index, centroid in enumerate(centroids):
            distance = numpy.linalg.norm(centroid - numpy.array(feature))
            heappush(clusters_heap, (distance, cluster_index))
        return nsmallest(n_closest, clusters_heap)

    @staticmethod
    def get_and_print_item_info(item_id, info_type):
        """
        Prints item info given item id
        :param item_id: input item id
        :param info_type: type of item (recommended, input,
        recommended base line, etc)
        """
        cur.execute(
            "SELECT listing_title, listing_description, "
            "category_l3_name_en FROM samples_train WHERE item_id=%s;",
            (item_id,))
        res = cur.fetchone()
        recomended_title, recomended_description, \
            recomended_category = res
        print(
            "%s info: %s %s %s %s" %
            (info_type, item_id, recomended_title, recomended_description,
             recomended_category))

    def get_recommended_candidates_base_line(self, category):
        """
        Base line recommendation: return n random items from same category as
        input category
        :param category: input category
        :return: n recommended candidates (n is defined at
        self.top_recommendations)
        """
        cur.execute(
            "SELECT item_id FROM samples_train WHERE category_l3_name_en=%s"
            " OFFSET RANDOM() * (SELECT COUNT(*) FROM "
            "samples_train WHERE category_l3_name_en=%s) LIMIT %s;",
            (category, category, self.top_recommendations))
        res = cur.fetchmany(self.top_recommendations)
        candidates_to_recommend_base_line = []
        for row in res:
            item_id, = row
            candidates_to_recommend_base_line.append((None, item_id))
        return candidates_to_recommend_base_line

    def get_recommended_candidates(self, vector_feature):
        """
        Recommendation: given vector feature return items from top n closest
        clusters as candidates to recommend
        :param vector_feature: input vector feature
        :return: recommended candidates from n closest clusters (n is defined at
        self.top_clusters)
        """
        n_closest_clusters = self.predict_cluster(
            vector_feature, self.centroids, self.top_clusters)
        candidates_to_recommend = []
        for distance, cluster_index in n_closest_clusters:
            for candidate in self.merged_clusters[cluster_index]:
                cur.execute(
                    "SELECT vector_feature FROM samples_train "
                    "WHERE item_id=%s;" % candidate)
                res_candidate = cur.fetchone()
                vector_feature_candidate, = res_candidate
                candidates_to_recommend.append(
                    (numpy.linalg.norm(
                        numpy.array(vector_feature) -
                        numpy.array(vector_feature_candidate)),
                     candidate))
        return candidates_to_recommend

    def get_recommended_candidates_for_test_data(self):
        """
        Gets recommedations and base line recommendations for test data
        """
        db_query = "SELECT item_id, listing_title, listing_description, " \
                   "category_l3_name_en, vector_feature FROM samples_test;"
        cur.execute(db_query)
        res = cur.fetchall()
        for row in res:
            item_id, title, description, category, vector_feature = row
            candidates_to_recommend = self.get_recommended_candidates(
                vector_feature)
            candidates_to_recommend_base_line = \
                self.get_recommended_candidates_base_line(category)
            print("***********************************************************"
                  "***********************************************************")
            print("Item info: %s %s %s %s" %
                  (item_id, title, description, category))
            for _, recomended_id in candidates_to_recommend[
                    :self.top_recommendations]:
                self.get_and_print_item_info(recomended_id, "Recommeded")
            for _, recomended_id_base_line in candidates_to_recommend_base_line:
                self.get_and_print_item_info(
                    recomended_id_base_line, "Recommeded base line")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    path_to_result_clusters = os.path.join(
        base_path, "result_clusters.pickle")
    recommender = Recommender(path_to_result_clusters, 5, 5)
    recommender.get_recommended_candidates_for_test_data()