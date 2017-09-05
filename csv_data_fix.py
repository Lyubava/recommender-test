#!/usr/bin/python3
# -*-coding: utf-8 -*-

import argparse
import csv
import os


class DataReaderFix(object):
    """
    Special class to fix corrupted csv file
    """
    def __init__(self, data_base_path):
        self.data_base_path = data_base_path
        self.sample_listings_path = os.path.join(
            self.data_base_path, "za_sample_listings_incl_cat.csv")
        self.saved_line = []

    def read_scv(self, csv_filename, delimiter, read_line):
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
            fout = open(os.path.join(
                self.data_base_path, "za_sample_listings_incl_cat_fixed.csv"),
                "w")
            writer = csv.writer(fout, delimiter=delimiter)
            next(fin)
            for line in reader:
                getattr(self, read_line)(line, writer)

    def read_samples_line_fix(self, line, writer):
        """
        Function to process one line of csv file
        :param line: line of csv file
        :param writer: file to save fixed lines
        :return:
        """
        try:
            _, item_id, seller_id, listing_title, listing_description, \
                listing_price, category_sk, category_l1_name_en, \
                category_l2_name_en, category_l3_name_en, listing_latitude, \
                listing_longitude = line
            writer.writerow(line)
        except:
            self.saved_line.extend(line)
        if self.saved_line:
            if len(self.saved_line) == 13 and "" in self.saved_line:
                new_line = [elem for elem in self.saved_line if elem]
                writer.writerow(new_line)
                self.saved_line = []
            elif (self.saved_line[0] == "227153" or
                  self.saved_line[0] == "307934" or
                  self.saved_line[0] == "485283") and \
                    len(self.saved_line) == 13:
                new_line = []
                for ind, elem in enumerate(self.saved_line):
                    if ind == 5:
                        new_elem = self.saved_line[4] + elem
                        new_line.append(new_elem)
                    elif ind == 4:
                        pass
                    else:
                        new_line.append(elem)
                writer.writerow(new_line)
                self.saved_line = []
            elif self.saved_line[0] == "299779" and len(self.saved_line) >= 12:
                new_line = []
                for ind, elem in enumerate(self.saved_line):
                    if ind == 9:
                        new_elem = ""
                        for index in range(4, 9):
                            new_elem += self.saved_line[index]
                        new_elem += elem
                        new_line.append(new_elem)
                    elif ind in range(4, 9):
                        pass
                    else:
                        new_line.append(elem)
                writer.writerow(new_line)
                self.saved_line = []
            elif len(self.saved_line) >= 12:
                import pdb
                pdb.set_trace()

    def read_samples(self):
        self.read_scv(
            self.sample_listings_path, ",", "read_samples_line_fix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--base-path", type=str, help="Path to directory with scv data",
        default=os.path.dirname(os.path.realpath(__file__)))
    args = parser.parse_args()
    base_path = args.base_path
    data_reader = DataReaderFix(base_path)
    data_reader.read_samples()