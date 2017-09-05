#!/usr/bin/python3
# -*-coding: utf-8 -*-

import copy
import os
import pickle
import math
import random
import numpy
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity

con = psycopg2.connect(
    user="postgres", host="localhost", dbname="olx_data", password="postgres")
con.autocommit = True
cur = con.cursor()


class ClustersModel(object):
    """
    Creates base clusters from queries, splits clusters with bisect K-means,
    merges close clusters
    """
    def __init__(self):
        self.query_count_threshold = 100
        self.clusters_for_bisect = None
        self.intersection_threshold = 0.8
        self.min_len_cluster = 17
        self.max_iterations = 100
        self.max_epoch = 10
        self.min_gain = 0.1

    @staticmethod
    def get_centroid(cluster):
        """
        Calculates centroid of cluster
        :param cluster: input cluster
        :return: centroid of input cluster
        """
        return numpy.mean(cluster, 0)

    @staticmethod
    def get_items_per_query(query, data_set):
        """
        Creates one base cluster from query: finds close items for query from
        database
        :param query: query
        :param data_set: type of data set (train or test)
        :return: cluster with features, corresponded item ids for cluster
        """
        db_query = "SELECT item_id, vector_feature FROM samples_%s WHERE " \
                   "listing_title LIKE %%s" % data_set
        cur.execute(db_query, (query,))
        res = cur.fetchall()
        query_cluster = []
        item_ids = []
        for row in res:
            item_id, vector_feature = row
            if vector_feature and item_id not in item_ids:
                query_cluster.append(vector_feature)
                item_ids.append(item_id)
        return query_cluster, item_ids

    @staticmethod
    def compute_intra_similarity_current(cluster_current):
        """
        Computes intra-cluster similarity for cluster
        :param cluster_current: input cluster
        :return: intra similarity
        """
        sum_cosine = sum(sum(
            cosine_similarity(cluster_current, cluster_current)))
        sq_root_sum_cosine = math.sqrt(sum_cosine)
        return sq_root_sum_cosine

    @staticmethod
    def get_all_items(clusters):
        """
        Unites all items from cluster into one global cluster
        :param clusters: input clusters
        :return: global cluster
        """
        all_items = []
        for cluster in clusters:
            for item in cluster:
                all_items.append(item)
        return all_items

    def get_error(self, cluster_current, all_items):
        """
        Computes inter_distortion/intra_similarity for cluster
        :param cluster_current: input cluster
        :param all_items: global cluster
        :return:
        """
        return (self.compute_inter_distortion_current(
            cluster_current, all_items) /
            self.compute_intra_similarity_current(cluster_current))

    def compute_inter_distortion_current(self, cluster_current, all_items):
        """
        Computes inter-cluster distortion for cluster
        :param cluster_current: input cluster
        :param all_items: one global cluster with all items
        :return: inter distortion
        """
        sum_cosine = sum(sum(cosine_similarity(cluster_current, all_items)))
        sq_root_sum_cosine = self.compute_intra_similarity_current(
            cluster_current)
        return (len(cluster_current) * sum_cosine) / sq_root_sum_cosine

    def get_sse(self, data):
        """
        Computes sum squared error
        :param data: input data
        :return: sum squared error
        """
        centroid = self.get_centroid(data)
        return numpy.sum(numpy.linalg.norm(data - centroid, 2, 1))

    def get_queries(self):
        """
        Filters queries for duplicates and frequency
        :return: filtered queries
        """
        cur.execute(
            "SELECT count, query FROM queries;")
        res = cur.fetchall()
        queries = {}
        for row in res:
            count, query = row
            if query not in queries:
                queries[query] = count
            else:
                queries[query] += count
        filtered_queries = [
            query for query, count in queries.items() if
            count > self.query_count_threshold]
        return filtered_queries

    def set_base_clusters(self, data_set):
        """
        Creates base clusters from frequent queries
        :param data_set: type of data set (train or test)
        :return: base clusters, corresponded item ids for base clusters
        """
        query_clusters = []
        cluster_items = []
        for query in self.get_queries():
            print("Get cluster for query %s" % query)
            query_cluster, item_ids = self.get_items_per_query(query, data_set)
            print(query_cluster)
            if query_cluster:
                query_clusters.append(query_cluster)
                cluster_items.append(item_ids)
        return query_clusters, cluster_items

    def kmeans_fit(self, data, data_items, clusters_number):
        """
        Basic k means. Split input data to clusters_number clusters
        :param data: input data
        :param data_items: corresponded item ids to input data
        :param clusters_number: output clusters number
        :return: output clusters, corresponded item ids to output clusters
        """
        min_error_combined = numpy.inf
        current_data = numpy.matrix(data)
        current_clusters = []
        current_clusters_items = []
        for epoch in range(self.max_epoch):
            iteration = 0
            error_combined_prev = numpy.inf
            centroids = random.sample(
                numpy.unique(data, axis=0).tolist(), clusters_number)
            while True:
                iteration += 1
                clusters = [None] * clusters_number
                clusters_indicies = [None] * clusters_number
                for item_index in range(current_data.shape[0]):
                    current_item = current_data[item_index]
                    cluster_index = numpy.argmin(numpy.linalg.norm(
                        current_item - centroids, 2, 1))
                    item = data_items[item_index]
                    if clusters[cluster_index] is None:
                        clusters[cluster_index] = [current_item.tolist()[0]]
                    else:
                        clusters[cluster_index].append(current_item.tolist()[0])
                    if clusters_indicies[cluster_index] is None:
                        clusters_indicies[cluster_index] = [item]
                    else:
                        clusters_indicies[cluster_index].append(item)
                if iteration >= self.max_iterations:
                    break
                for cluster_index in range(clusters_number):
                    centroids[cluster_index] = self.get_centroid(
                        clusters[cluster_index])
                error_combined = numpy.sum(
                    [self.get_sse(clusters[cluster_index]) for cluster_index
                     in range(clusters_number)])
                gain = error_combined_prev - error_combined
                print(
                    "Previos %s Current %s Gain %s Minimum %s" %
                    (error_combined_prev, error_combined, gain,
                     min_error_combined))
                if gain < self.min_gain:
                    if error_combined < min_error_combined:
                        min_error_combined, current_clusters, \
                            current_clusters_items = error_combined, clusters, \
                            clusters_indicies
                    break
                else:
                    error_combined_prev = error_combined
        print(current_clusters_items)
        return current_clusters, current_clusters_items

    def bisect_kmeans(self, base_clusters, clusters_items):
        """
        Splits base clusters to smaller ones
        :param base_clusters: input base clusters
        :param clusters_items: corresponded item ids to input clusters
        :return: output clusters, corresponded item ids to output clusters
        """
        clusters_for_bisect = base_clusters
        clusters_result = []
        clusters_result_items = []
        clusters_items_current = copy.deepcopy(clusters_items)
        while True:
            clusters_good = []
            clusters_for_bisect_ = []
            clusters_for_bisect_items = []
            clusters_good_items = []
            for index, cluster in enumerate(clusters_for_bisect):
                items = clusters_items_current[index]
                if len(cluster) > self.min_len_cluster:
                    clusters_for_bisect_.append(cluster)
                    clusters_for_bisect_items.append(items)
                else:
                    clusters_good.append(cluster)
                    clusters_good_items.append(items)
            clusters_result.extend(clusters_good)
            clusters_result_items.extend(clusters_good_items)
            clusters_for_bisect = copy.deepcopy(clusters_for_bisect_)
            len_bisect = len(clusters_for_bisect)
            print("Clusters to bisect number: %s" % len_bisect)
            all_items = self.get_all_items(clusters_for_bisect)
            errors_bisect = [
                self.get_error(cluster, all_items) for cluster in
                clusters_for_bisect]
            if len_bisect < 1:
                break
            combined_error_prev_bisect = numpy.sum(errors_bisect)
            index_cluster = numpy.argmax(
                errors_bisect)
            cluster_to_bisect_items = clusters_for_bisect_items.pop(
                index_cluster)
            cluster_max_error = clusters_for_bisect.pop(
                index_cluster)
            cluster_base = numpy.array(cluster_max_error)
            assert len(cluster_max_error) == len(cluster_to_bisect_items)
            print("Cluster to bisect len: %s" % len(cluster_max_error))
            clusters_number = 2
            unique_features = numpy.unique(cluster_base, axis=0).tolist()
            if len(unique_features) >= clusters_number:
                bisected_clusters, bisected_clusters_items = self.kmeans_fit(
                    cluster_base, cluster_to_bisect_items, clusters_number)
                for index, cluster in enumerate(bisected_clusters):
                    clusters_for_bisect.append(cluster)
                    cluster_items = bisected_clusters_items[index]
                    clusters_for_bisect_items.append(cluster_items)
            else:
                clusters_result.append(cluster_base)
                clusters_result_items.append(cluster_to_bisect_items)
            clusters_items_current = copy.deepcopy(clusters_for_bisect_items)
            print("Current error: %s" % (combined_error_prev_bisect/len_bisect))
            if len(clusters_for_bisect) < 1:
                break
        print("Previous clusters number: %s" % len(base_clusters))
        print("Current clusters number: %s" % len(clusters_result))
        return clusters_result, clusters_result_items

    def compare_clusters(
            self, cluster_one, cluster_two, cluster_one_index,
            cluster_two_index):
        """
        Compares two input clusters
        :param cluster_one: first input cluster items
        :param cluster_two: second input cluster items
        :param cluster_one_index: first input cluster index
        :param cluster_two_index: second input cluster index
        :return: (max cluster index, min cluster index), relation between
        clusters
        """
        clusters_intersection = list(set(cluster_one).intersection(cluster_two))
        len_cluster_one = len(cluster_one)
        len_cluster_two = len(cluster_two)
        if len_cluster_one < len_cluster_two:
            min_len = len_cluster_one
            max_cluster_index = cluster_two_index
            min_cluster_index = cluster_one_index
        else:
            min_len = len_cluster_two
            max_cluster_index = cluster_one_index
            min_cluster_index = cluster_two_index
        len_intersection = len(clusters_intersection)
        if self.intersection_threshold * min_len < len_intersection < min_len:
            return [max_cluster_index, min_cluster_index], "absorb"
        elif len_intersection >= min_len:
            return [max_cluster_index, min_cluster_index], "parent-child"
        else:
            return [max_cluster_index, min_cluster_index], "none"

    def merge_clusters(self, granual_clusters_items):
        """
        Merges close clusters
        :param granual_clusters_items: input cluster items
        :return: merged clusters items, hierarchy with parent-child
        relations between clusters indicies
        """
        hierarchy = {}
        clusters = []
        print("Will merge clusters")
        len_granual_clusters_items = len(granual_clusters_items)
        indicies_to_delete = []
        print("Previous clusters number: %s" % len_granual_clusters_items)
        for index_one, cluster_one in enumerate(granual_clusters_items):
            for index_two in range(index_one + 1, len_granual_clusters_items):
                cluster_two = granual_clusters_items[index_two]
                merged_clusters_indicies, relation = self.compare_clusters(
                    cluster_one, cluster_two, index_one, index_two)
                index_1 = merged_clusters_indicies[0]
                index_2 = merged_clusters_indicies[1]
                if relation == "absorb":
                    indicies_to_delete.append(index_2)
                elif relation == "parent-child":
                    if index_1 not in hierarchy:
                        hierarchy[index_1] = [index_2]
                    else:
                        hierarchy[index_1].append(index_2)
                else:
                    if index_1 not in hierarchy:
                        hierarchy[index_1] = None
                    if index_2 not in hierarchy:
                        hierarchy[index_2] = None
        for cluster_index in range(len_granual_clusters_items):
            if cluster_index in indicies_to_delete:
                continue
            clusters.append(granual_clusters_items[cluster_index])
        print("Current clusters number: %s" % len(clusters))
        return clusters, hierarchy

    def get_clusters_centroids(self, clusters_items, data_set):
        """
        Calculates centroids of clusters given clusters item ids
        :param clusters_items: input clusters items ids
        :param data_set: type of data set (train or test)
        :return: centroids for clusters
        """
        print("Will get clusters centroids")
        centroids = []
        for cluster in clusters_items:
            print("Calculate centroid for cluster %s" % str(cluster))
            vector_cluster = []
            for item_id in cluster:
                db_query = "SELECT vector_feature FROM samples_%s WHERE " \
                           "item_id=%%s" % data_set
                cur.execute(db_query, (item_id,))
                res = cur.fetchone()
                vector_feature, = res
                vector_cluster.append(vector_feature)
            centroid = self.get_centroid(vector_cluster)
            centroids.append(centroid)
        return centroids

    def run(self):
        """
        Calculate base clusters, splits it with k means bisect, merges
        clusters items and calculates centroids for merged clusters
        :return:
        """
        set_of_data = "train"
        base_path = os.path.dirname(os.path.realpath(__file__))
        to_base_clusters = os.path.join(base_path, "base_clusters.pickle")
        to_result_clusters = os.path.join(
            base_path, "result_clusters.pickle")
        baseclusters, cluster_item = self.set_base_clusters(
            data_set=set_of_data)
        with open(to_base_clusters, "wb") as fout:
            pickle.dump(baseclusters, fout)
            pickle.dump(cluster_item, fout)
        _, result_items = self.bisect_kmeans(
            baseclusters, cluster_item)
        merged_clusters, hierarchy = self.merge_clusters(result_items)
        result_centroids = self.get_clusters_centroids(
            merged_clusters, set_of_data)
        with open(to_result_clusters, "wb") as fout:
            pickle.dump(merged_clusters, fout)
            pickle.dump(hierarchy, fout)
            pickle.dump(result_centroids, fout)
        return merged_clusters, result_centroids, hierarchy


if __name__ == "__main__":
    clusters_model = ClustersModel()
    clusters_model.run()