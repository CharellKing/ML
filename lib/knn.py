#!/usr/bin/python
# -*-coding:utf-8-*-

import sys

class KNN(object):
    def __init__(self, k, matrix=None, matrix_labels=None, min_maxs=None):
	self.matrix = matrix
	self.matrix_labels = matrix_labels
	self.min_maxs = min_maxs
	self.k = k

    def auto_normal_field(self, val, col_index):
    	if self.min_maxs:
	    return (val - self.min_maxs[col_index][0]) / (self.min_maxs[col_index][1] - self.min_maxs[col_index][0])
	return val

    def auto_normal_fields(self, fields):
	if self.min_maxs:
	    for i in xrange(len(fields)):
            	fields[i] = self.auto_normal_field(fields[i], i);
	return fields

    def load(self, file_path, sep, min_maxs = None):
        self.matrix, self.matrix_labels = [], []
	self.min_maxs = min_maxs
	with open(file_path, "r") as f:
	    while True:
		line = f.readline()
		if not line:
		    break
		fields = line.strip().split(sep)
		self.matrix_labels.append(fields[-1])
		self.matrix.append(self.auto_normal_fields([float(field) for field in fields[0:len(fields)-1]]))

    def square_distance(self, fields_a, fields_b):
	ret = 0
	for i in xrange(len(fields_a)):
	    val = fields_a[i] - fields_b[i]
	    ret += (val * val)
	return ret

    def classify(self, fields):
	distance_labels = []
	for i in xrange(len(self.matrix)):
	    sd = self.square_distance(self.matrix[i], fields)
	    distance_labels.append([sd, self.matrix_labels[i]])
	distance_labels = sorted(distance_labels, key=lambda distance_label:distance_label[0])
	label_counts = {}
	for i in xrange(self.k):
	    if distance_labels[i][1] not in label_counts:
		label_counts[distance_labels[i][1]] = 0
	    label_counts[distance_labels[i][1]] += 1

        min_distance, min_label = sys.float_info.max, None
        for label_count in label_counts.items():
            if label_count[1] < min_distance:
                min_label = label_count[0]
        return min_label

    def multiple_classify(self, matrix):
	labels = []
	for fields in matrix:
	    labels.append(self.classify(fields))
	return labels

    def test_knn_from_data(self, matrix, matrix_labels):
	new_matrix_labels = self.multiple_classify(matrix)
	success_samples, failed_samples = 0, 0
	for i in xrange(len(matrix_labels)):
	    if new_matrix_labels[i] == matrix_labels[i]:
		success_samples += 1
	    else:
		failed_samples += 1
	print "success:%0.2f%%, failed:%0.2f%%" % (success_samples * 100.0 / len(matrix_labels), failed_samples * 100.0 / len(matrix_labels))

    def test_knn_from_file(self, file_path, sep):
	matrix_labels, matrix = [], []
	with open(file_path, "r") as f:
	    while True:
		line = f.readline()
		if not line:
		    break
		fields = line.strip().split(sep)
		matrix_labels.append(fields[-1])
		matrix.append(self.auto_normal_fields([float(field) for field in fields[0:len(fields)-1]]))
        self.test_knn_from_data(matrix, matrix_labels)

def main():
    knn = KNN(3)
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    knn.load("%s/../data/iris/iris.data" % (cur_dir), ",", [[4,8],[2,5],[1,7],[0,3]])
    knn.test_knn_from_file("%s/../data/iris/bezdekIris.data" % (cur_dir), ",")

if __name__ == "__main__":
    import sys
    import os
    reload(sys)
    sys.setdefaultencoding("utf-8")

    main()
