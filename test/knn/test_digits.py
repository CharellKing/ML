#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import sys

cur_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0, "%s/../../lib/" % (cur_dir))
from knn import KNN

def number_to_files(data_dir):
    number_paths = []
    for file_name in os.listdir(data_dir):
        number = file_name.split("_")[0]
        path = "%s/%s" % (data_dir, file_name)
        number_paths.append([number, path])
    return number_paths

def get_fields(file_path):
    with open(file_path, "r") as f:
        fields = []
        while True:
            line = f.readline()
            if not line:
                break
            fields.extend([int(ch) for ch in list(line.strip())])
    return fields

def get_matrix_labels(data_dir):
    matrix, labels = [], []
    number_paths = number_to_files(data_dir)
    for number, file_path in number_paths:
        matrix.append(get_fields(file_path))
        labels.append(number)
    return matrix, labels

def main():
    training_dir = os.path.realpath("%s/../../data/digits/trainingDigits/" % (cur_dir))
    training_matrix, training_labels = get_matrix_labels(training_dir)
    knn = KNN(3, training_matrix, training_labels)

    test_dir = os.path.realpath("%s/../../data/digits/testDigits/" % (cur_dir))
    test_matrix, test_labels = get_matrix_labels(test_dir)
    knn.test_knn_from_data(test_matrix, test_labels)

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    main()
