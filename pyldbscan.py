import sys
import numpy as np
import math
import random
import time

class LDBSCAN:

    def __init__(self):
        self.records = [] #arraylist
        self.qs = Quicksort()
        self.lrd = [] #double
        self.lof = [] #double
        self.belong = None #int
        self.knn_dist = None #double [][]
        self.knn_seq = None #int [][]
        self.clustering_sequence = [] # arraylist

    def runLDBSCAN(self, filename, MinPts_LOF, MinPts_LDBSCAN, core_lof, pct):
        self.loadData(filename)
        start_time = time.time()
        self.calculateDistance(max(MinPts_LOF, MinPts_LDBSCAN))
        end_time = time.time()
        print "Time elapsed: ", end_time - start_time, "s"
        self.calculateLRD(MinPts_LOF);
        self.calculateLOF(MinPts_LOF);
        self.clustering(MinPts_LDBSCAN, core_lof, pct);

    def loadData(self,filename):
        f = file(filename)
        for data in f.readlines():
            record = []
            attributes = data.split(",")
            #this part can be optimized as record = data.split(",")
            for i in xrange(len(attributes)):
                record.append(float(attributes[i]))
            self.records.append(record)

        f.close()

    def calculateDistance(self,MinPts):
        n = len(self.records) # number of records
        m = len(self.records[0]) # number of attributes
        self.knn_dist = np.zeros(shape=(n,MinPts+1))
        self.knn_seq = np.zeros(shape=(n,MinPts+1))
        for i in xrange(n):
            first_object = self.records[i]
            distance = [0] * n
            for j in xrange(n):
                second_object = self.records[j]
                for k in xrange(m):
                    distance[j] += (float(first_object[k]) - float(second_object[k])) * (float(first_object[k]) - float(second_object[k]))
                distance[j] = math.sqrt(distance[j])
            sequence = self.getTopKSequenceOptimized(distance, MinPts + 1, False)
            for j in xrange(len(sequence)):
                self.knn_dist[i][j] = distance[sequence[j]]
                self.knn_seq[i][j] = int(sequence[j])

    def calculateLRD(self,MinPts_LOF):
        self.lrd = [0] * len(self.records)
        for i in xrange(len(self.records)):
            reach_dist = 0
            for j in xrange(MinPts_LOF):
                if (self.knn_dist[i][j] > self.knn_dist[int(self.knn_seq[i][j])][MinPts_LOF]):
                    reach_dist += self.knn_dist[i][j]
                else:
                    reach_dist += self.knn_dist[int(self.knn_seq[i][j])][MinPts_LOF]
            self.lrd[i] = float(MinPts_LOF/reach_dist)

    def calculateLOF(self,MinPts_LOF):
        self.lof = [0] * len(self.records)
        for i in xrange(len(self.records)):
            lrd_sum = float(0)
            for j in xrange(MinPts_LOF):
                lrd_sum += self.lrd[int(self.knn_seq[i][j])]
            self.lof[i] = float(lrd_sum) / float((MinPts_LOF * self.lrd[i]))

    def clustering(self, MinPts_LDBSCAN, core_lof, pct):
        id_seq = [0] * len(self.records)
        self.belong = [0] * len(self.records)
        for i in xrange(len(self.records)):
            id_seq[i] = i

        sequence = self.qs.sort(self.lof, id_seq, True)
        unassigned_objs_id = []
        for i in xrange(len(sequence)):
            unassigned_objs_id.append(sequence[i])
        cluster_id = 0
        while (unassigned_objs_id):
            obj_id = int(unassigned_objs_id[0])
            if (self.lof[obj_id] < core_lof):
                cluster_id += 1
                tempList = []
                unassigned_objs_id.pop(unassigned_objs_id.index(int(obj_id)))
                tempList.append(obj_id)
                while (tempList):
                    obj_id = int(tempList[0])
                    tempList.pop(0)
                    self.clustering_sequence.append(obj_id)
                    self.belong[obj_id] = cluster_id
                    lrd_ub = float(self.lrd[obj_id] * (1 + pct))
                    lrd_lb = float(self.lrd[obj_id] / (1 + pct))
                    for i in xrange(MinPts_LDBSCAN):
                        if (int(self.knn_seq[obj_id][i]) in unassigned_objs_id) \
                            and (self.lrd[int(self.knn_seq[obj_id][i])] < lrd_ub) \
                            and (self.lrd[int(self.knn_seq[obj_id][i])] > lrd_lb) \
                            and (int(self.knn_seq[obj_id][i]) not in tempList):
                                unassigned_objs_id.pop(unassigned_objs_id.index(int(self.knn_seq[obj_id][i])))
                                tempList.append(self.knn_seq[obj_id][i])


            else:
                self.belong[obj_id] = 0
                unassigned_objs_id.pop(unassigned_objs_id.index(int(obj_id)))
        print "id,",
        for i in xrange(len(self.records[0])):
            print "attri" + str(i+1) + ",",

        print "LRD,LOF,clusterID"
        for i in xrange(len(self.belong)):
            print str(i), ",",
            record = self.records[i]
            for j in xrange(len(record)):
                print record[j], ",",
            print self.lrd[i],"," ,self.lof[i],",",self.belong[i]

    #returns a list, original_value is a list, get_max_value is a boolean
    def getTopKSequence(self, original_value, k, get_max_value):
        total_length = len(original_value)
        top_k_sequence = [0] * k
        top_k_value = [0] * k
        if (k == 1):
            cur_value = original_value[0]
            cur_seq = 0
            if (get_max_value):
                for i in xrange(total_length):
                    if (cur_value < original_value[i]):
                        cur_value = original_value[i]
                        cur_seq = i
            else:
                for i in xrange(total_length):
                    if (cur_value > original_value[i]):
                        cur_value = original_value[i]
                        cur_seq = i
            top_k_sequence[0] = cur_seq
        elif (k <= total_length):
            if (get_max_value):
                for i in xrange(k):
                    top_k_value[i] = float("-inf")
                for i in xrange(total_length):
                    position = k
                    for j in xrange(k-1):
                        if (top_k_value[k - 1 - j] < original_value[i]):
                            top_k_value[k - 1 - j] = top_k_value[k - 2 - j]
                            top_k_sequence[k - 1 - j] = top_k_sequence[k - 2 - j]
                            position -= 1
                        else:
                            break
                    if (position < k):
                        if (position == 1 and top_k_value[0] < original_value[i]):
                            position -= 1
                        top_k_value[position] = original_value[i]
                        top_k_sequence[position] = i
            else:
                for i in xrange(k):
                    top_k_value[i] = float("inf")
                for i in xrange(total_length):
                    position = k
                    for j in xrange(k-1):
                        if (top_k_value[k - 1 - j] > original_value[i]):
                            top_k_value[k - 1 - j] = top_k_value[k - 2 - j]
                            top_k_sequence[k - 1 - j] = top_k_sequence[k - 2 - j]
                            position -= 1
                        else:
                            break
                    if (position < k):
                        if (position == 1 and top_k_value[0] > original_value[i]):
                            position -= 1
                        top_k_value[position] = original_value[i]
                        top_k_sequence[position] = i
        else:
            print "Error: the k is greater than the array length!"

        return top_k_sequence

    def getTopKSequenceOptimized(self, original_value, k, get_max_value):
        return [i[0] for i in sorted(enumerate(original_value), key=lambda x:x[1])[:k]]



class Quicksort:

    #array, sequence, breakRank is a list, i and j are integers
    def swap(self, array, sequence, breakRank, i, j):
        tmp = array[i]
        array[i] = array[j]
        array[j] = tmp
        tmp_seq = sequence[i]
        sequence[i] = sequence[j]
        sequence[j] = tmp_seq
        tmp_seq = breakRank[i]
        breakRank[i] = breakRank[j]
        breakRank[j] = tmp_seq

    #array, sequence, breakRank are list, begin and are integers
    def partition(self, array, sequence, breakRank, begin, end):
        index = len(array) #added manually
        while index == len(array): # added manually
            index = begin + random.randint(0, end - begin + 1)
        pivot = array[index]
        pivotrank = breakRank[index]
        self.swap(array, sequence, breakRank, index, end)
        index = begin
        for i in xrange(begin, end):
            if (array[i] > pivot or (array[i] == pivot and breakRank[i] < pivotrank)):
                self.swap(array, sequence, breakRank, index, i)
                index += 1

        self.swap(array, sequence, breakRank, index, end)
        return index

    #array, sequence and breakRank are list, begin and end are integers
    #return is a lits
    def qsort(self, array, sequence, breakRank, begin, end):
        if (end > begin):
            index = self.partition(array, sequence, breakRank, begin, end)
            self.qsort(array, sequence, breakRank, begin, index - 1)
            self.qsort(array, sequence, breakRank, index + 1, end)

    #org_array, breakRank is list, ascending is Boolean
    #return is a list
    def sort(self, org_array, breakRank, ascending):
        sequence = [0] * len(org_array)
        reverse_sequence = [0] * len(org_array)
        array = [0] * len(org_array)
        for i in xrange(len(org_array)):
            sequence[i] = i
            array[i] = org_array[i]

        self.qsort(array, sequence, breakRank, 0, len(array) - 1)
        for i in xrange(len(org_array)):
            reverse_sequence[i] = sequence[len(org_array) - 1 - i]

        if (ascending):
            return reverse_sequence
        else:
            return sequence


if __name__ == "__main__":
    tmp = LDBSCAN()
    if len(sys.argv) != 5:
        tmp.runLDBSCAN("sampleData.csv", 15, 10, 2, 0.2)
    else:
        tmp.runLDBSCAN(sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))