#!/usr/bin/python3
# -*-coding: utf-8 -*-

"""
Script to read data from csv and save it to PostgreSQL DB
Execute before proceed:
sudo -u postgres psql postgres
alter user postgres with password 'postgres';

Data was corrupt. Please make sure that file is fixed before proceed
(run csv_data_fix.py)
"""

import argparse
import csv
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sklearn.model_selection import train_test_split


con = psycopg2.connect(
    user="postgres", host="localhost", dbname="postgres", password="postgres")
database_name = "olx_data"

con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = con.cursor()
cur.execute("DROP DATABASE IF EXISTS %s;" % database_name)
cur.execute("CREATE DATABASE %s;" % database_name)

con = psycopg2.connect(
    user="postgres", host="localhost", dbname="olx_data", password="postgres")
con.autocommit = True
cur = con.cursor()


class DataReader(object):
    """
    Reads data from csv files and adds it to PostgreSQL Database
    """
    def __init__(self, data_base_path):
        self.data_base_path = data_base_path
        self.queries_path = os.path.join(
            self.data_base_path, "za_queries_sample.csv")
        self.sample_listings_path = os.path.join(
            self.data_base_path, "za_sample_listings_incl_cat.csv")
        self.test_items = []

    @staticmethod
    def convert_to_float(item):
        if item:
            item = float(item)
        else:
            item = None
        return item

    @staticmethod
    def read_queries_line(line):
        """
        Processes one line fron csv queries file
        :param line: csv file line
        """
        _, query, count = line
        print(query)
        cur.execute(
            "INSERT INTO queries (query, count) VALUES (%s, %s)",
            (query, int(count)))

    def read_samples_line(self, line):
        """
        Processes one line fron csv samples file
        :param line: csv file line
        """
        _, item_id, seller_id, listing_title, listing_description, \
            listing_price, category_sk, category_l1_name_en, \
            category_l2_name_en, category_l3_name_en, listing_latitude, \
            listing_longitude = line
        if item_id in self.test_items:
            print("*********************Test********************: %s" % item_id)
            cur.execute(
                "INSERT INTO samples_test (item_id, seller_id, listing_title, "
                "listing_description, listing_price, category_sk, "
                "category_l1_name_en, category_l2_name_en, "
                "category_l3_name_en, listing_latitude, listing_longitude) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (int(item_id), int(seller_id), listing_title,
                 listing_description, self.convert_to_float(listing_price),
                 category_sk, category_l1_name_en, category_l2_name_en,
                 category_l3_name_en,
                 self.convert_to_float(listing_latitude),
                 self.convert_to_float(listing_longitude)))
        else:
            print("Train: %s" % item_id)
            cur.execute(
                "INSERT INTO samples_train (item_id, seller_id, listing_title, "
                "listing_description, listing_price, category_sk, "
                "category_l1_name_en, category_l2_name_en, "
                "category_l3_name_en, listing_latitude, listing_longitude) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (int(item_id), int(seller_id), listing_title,
                 listing_description, self.convert_to_float(listing_price),
                 category_sk, category_l1_name_en, category_l2_name_en,
                 category_l3_name_en,
                 self.convert_to_float(listing_latitude),
                 self.convert_to_float(listing_longitude)))

    def read_csv(self, csv_filename, delimiter, read_line):
        """
        Read csv file
        :param csv_filename: path to csv data
        :param delimiter: delimeter for csv file
        :param read_line: function to process one line of csv file
        """
        if not os.access(csv_filename, os.R_OK):
            raise OSError(
                "Can not read %s. There is no such file or you have no "
                "permission to read it" % csv_filename)
        with open(csv_filename, "r") as fin:
            reader = csv.reader(fin, delimiter=delimiter)
            next(fin)
            for line in reader:
                getattr(self, read_line)(line)

    def read_queries(self):
        """
        Reads queries data from scv. Appends it to PostgerSQL table
        """
        cur.execute("""
            DROP TABLE IF EXISTS queries;
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries(
                query character varying(500),
                count integer)
        """)
        self.read_csv(
            self.queries_path, ",", "read_queries_line")

    def get_test_train_split(self, csv_filename, delimiter, test_size):
        """
        Split data to train and test sets: get test items ids from csv file
        :param csv_filename: path to csv data
        :param delimiter: delimeter for csv file
        :param test_size: proportion of the dataset to include in the test
        split
        """
        with open(csv_filename, "r") as fin:
            reader = csv.reader(fin, delimiter=delimiter)
            row_count = sum(1 for _ in reader)
        _, self.test_items = train_test_split(
            [str(elem) for elem in list(range(row_count - 1))],
            test_size=test_size)

    def read_samples(self):
        """
        Reads samples data from scv. Appends it to PostgerSQL table
        """
        cur.execute("""
            DROP TABLE IF EXISTS samples_test;
        """)
        cur.execute("""
            DROP TABLE IF EXISTS samples_train;
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS samples_test (
                item_id bigint,
                seller_id bigint,
                listing_title character varying(500),
                listing_description text,
                listing_price double precision,
                category_sk character varying(100),
                category_l1_name_en character varying(100),
                category_l2_name_en character varying(100),
                category_l3_name_en character varying(100),
                listing_latitude double precision,
                listing_longitude double precision)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS samples_train (
                item_id bigint,
                seller_id bigint,
                listing_title character varying(500),
                listing_description text,
                listing_price double precision,
                category_sk character varying(100),
                category_l1_name_en character varying(100),
                category_l2_name_en character varying(100),
                category_l3_name_en character varying(100),
                listing_latitude double precision,
                listing_longitude double precision)
        """)
        self.get_test_train_split(self.sample_listings_path, ",", 0.2)
        self.read_csv(
            self.sample_listings_path, ",", "read_samples_line")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--base-path", type=str, help="Path to directory with csv data",
        default=os.path.dirname(os.path.realpath(__file__)))
    args = parser.parse_args()
    base_path = args.base_path
    data_reader = DataReader(base_path)
    data_reader.read_samples()
    data_reader.read_queries()