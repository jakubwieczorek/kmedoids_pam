# https://pbpython.com/categorical-encoding.html
# https://pl.wikipedia.org/wiki/Algorytm_PAM
import csv
import getopt
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json

class KMedoids:
    def __init__(self, n_cluster=2, max_iter=10, start_prob_dist=0.8, end_prob_dist=0.99):
        if start_prob_dist < 0 or start_prob_dist >= 1 or end_prob_dist < 0 or end_prob_dist >= 1 or start_prob_dist > end_prob_dist:
            raise ValueError('Invalid input')
        self.n_cluster = n_cluster
        self.max_iter = max_iter

        # filtering variables in __select_distant_medoid
        self.start_prob_dist = start_prob_dist
        self.end_prob_dist = end_prob_dist

        self.medoids = []
        self.clusters = {}
        self.current_distance = 0

        self.__data = None
        self.__rows = 0
        self.cluster_distances = {}

    def fit(self, data):
        self.__data = data
        self.__rows = len(self.__data)
        self.__initialize_medoids()
        self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
        self.__update_clusters()
        return self

    def __update_clusters(self):
        """
        (SWAP) While the cost of the configuration decreases
        """
        for i in range(self.max_iter):
            cluster_dist_with_new_medoids = self.__swap_and_recalculate_clusters()
            if self.__cost_decreases(cluster_dist_with_new_medoids):
                self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
            else:
                break

    def __cost_decreases(self, cluster_dist_with_new_medoids):
        """
        Check if distances of the new medoids in custers are smaller
        than the current one
        """
        old_dist = self.__calculate_distance_of_clusters()
        new_dist = self.__calculate_distance_of_clusters(cluster_dist_with_new_medoids)

        if old_dist > new_dist:
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True
        return False

    def __calculate_distance_of_clusters(self, cluster_dist=None):
        if cluster_dist is None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist

    def __swap_and_recalculate_clusters(self):
        cluster_dist = {}
        # For each medoid m
        for m in self.medoids:
            cost_changed = False

            # and for each non-medoid data point o
            for o in self.clusters[m]:
                if o != m:
                    # swap
                    cluster_list = list(self.clusters[m])
                    cluster_list[self.clusters[m].index(o)] = m
                    cluster_list[self.clusters[m].index(m)] = o

                    # compute the cost change
                    new_distance = self.calculate_cluster_distance(o, cluster_list)

                    # If the cost change is the current best, remember this m and o combination
                    if new_distance < self.cluster_distances[m]:
                        cluster_dist[o] = new_distance
                        cost_changed = True
                        break

            # If the cost change is the current best, remember this m and o combination
            if not cost_changed:
                cluster_dist[m] = self.cluster_distances[m]

        return cluster_dist

    def calculate_cluster_distance(self, m, cluster):
        """
        Calculate the distance of m to each o in the given cluster
        """
        distance = 0
        for o in cluster:
            distance += self.__calculate_distance(m, o)
        return distance / len(cluster)

    def __calculate_clusters(self, medoid_indexes):
        """
        1) Associate each data point to the closest medoid -> define clusters
        2) For each cluster calculate the distance from its medoid to all the cluster members
        3) Scale the distance to the amount of cluster members
        """

        # initialize
        clusters = {}
        cluster_distances = {}
        for medoid_idx in medoid_indexes:
            clusters[medoid_idx] = []
            cluster_distances[medoid_idx] = 0

        # define clusters and calculate distances
        for row in range(self.__rows):
            nearest_medoid, nearest_distance = self.__calculate_shortest_distance_to_medoid(row, medoid_indexes)
            cluster_distances[nearest_medoid] += nearest_distance
            clusters[nearest_medoid].append(row)

        # scale the distances
        for medoid_idx in medoid_indexes:
            cluster_distances[medoid_idx] /= len(clusters[medoid_idx])
        return clusters, cluster_distances

    def __calculate_shortest_distance_to_medoid(self, row_index, medoid_indexes):
        """
        Calculate the shortest distance from the particular row to all the medoids.
        Parameters medoids and row_indexes are indexes, not entries
        """
        min_distance = float('inf')
        current_m = None

        for m in medoid_indexes:
            current_distance = self.__calculate_distance(m, row_index)
            if current_distance < min_distance:
                min_distance = current_distance
                current_m = m
        return current_m, min_distance

    def __initialize_medoids(self):
        """
        Initialize: greedily select k of the n data points as the medoids to minimize the cost.
        First medoid is random. Next ones are chosen basing on the maximum distance from this medoid
        to the rest of the data.
        """
        self.medoids.append(random.randint(0, self.__rows - 1))
        while len(self.medoids) != self.n_cluster:
            self.medoids.append(self.__find_distant_medoid())

    def __find_distant_medoid(self):
        """
        Finds index of one of the distant medoids. For every row:
        1) Calculate the shortest distance from the set of all medoids
        2) Sort the distances from min to max
        3) Filter them to take into account only those from
            <start_prob_dist*len(distances_index); end_prob_dist(distances_index)>, for example
            for 20 rows, start_prob_dist == 0.8 and end_prob_dist == 0.8 -> <16; 18>
        4) Randomly choose from the above range one distance
        5) Return the index row, so the data row for this distance
        """
        distances = []  # contains the distances of each row entry from the medoid
        indices = []  # rows == indices, so 0, 1, 2, 3, etc.
        for row in range(self.__rows):
            indices.append(row)
            nearest_medoid, nearest_distance = self.__calculate_shortest_distance_to_medoid(row, self.medoids)
            distances.append(nearest_distance)
        distances_index = np.argsort(distances)  # distance indexes from min to max
        chosen_dist = self.__select_distant_medoid(distances_index)
        return indices[chosen_dist]

    def __select_distant_medoid(self, distances_index):
        """
        Takes one of the distant medoids index: start_index bases on the start_prob_dist, so
        for example start_prob_dist == 0.8, there are 20 distant indexes (sorted from min to max!,
        the further the more distant),
        then start_index == round(0.8*20) == 16, so it will take from 16th to the same for
        end_prob_dist
        """
        start_index = round(self.start_prob_dist * len(distances_index))
        end_index = round(self.end_prob_dist * (len(distances_index) - 1))
        return distances_index[random.randint(start_index, end_index)]

    def __calculate_distance(self, x1, x2):
        a = np.array(self.__data[x1])
        b = np.array(self.__data[x2])
        return np.linalg.norm(a - b)


def plot_graphs(data, k_medoids):
    colors = {0: 'b*', 1: 'g^', 2: 'ro', 3: 'c*', 4: 'm^', 5: 'yo', 6: 'ko', 7: 'w*'}
    index = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for key in k_medoids.clusters.keys():
        temp_data = k_medoids.clusters[key]
        x = [data[i][0] for i in temp_data]
        y = [data[i][1] for i in temp_data]
        ax[0].plot(x, y, colors[index])
        index += 1
    # plt.show()

    medoid_data_points = []
    for m in k_medoids.medoids:
        medoid_data_points.append(data[m])
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_ = [i[0] for i in medoid_data_points]
    y_ = [i[1] for i in medoid_data_points]

    ax[1].plot(x, y, 'yo')
    ax[1].plot(x_, y_, 'r*')

    # plt.plot(x, y, 'yo')
    # plt.plot(x_, y_, 'r*')
    plt.show()


def show_help():
    print('python KMedoids.py')
    print('-h, --help prints this help')
    print('-c, --clusters cluster amount')
    print('-i, --inputs csv input file name')
    print('-c, --delimiter csv delimiter (put it between "")')

def parse_args(argv):
    input_file_name = None
    clusters = None
    delimiter = None

    if len(argv) < 3:
        print('python KMedoids.py -i <input_file> -c <cluster_amount>')
        sys.exit(2)

    try:
        opts, args = getopt.getopt(argv, "hi:c:d:", ["help", "input=", "clusters=", "delimiter="])
    except getopt.GetoptError:
        print('python KMedoids.py -i <input_file> -c <cluster_amount>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--input'):
            input_file_name = arg
        elif opt in ('-c', '--clusters'):
            clusters = int(arg)
        elif opt in ('-d', '--delimiter'):
            delimiter = arg
        elif opt in ('-h', '--help'):
            show_help()
            sys.exit(2)

    return input_file_name, clusters, delimiter

def parse_csv(file_name, delimiter=";"):
    with open(file_name, "r") as csv_file:
        content = csv.reader(csv_file, delimiter=delimiter)

        data = []
        # avoid first column and row (descriptive cells)
        column_amount = len(next(content))
        dict = {}
        row_number = 0
        for row in content:
            dict[row_number] = row[0]
            row_number += 1
            data.append([float(i) for i in row[1:column_amount]])

        return data, dict

def create_output_file(dict, a_clusters, input_file_name):
    clusters = {}
    for key, value in a_clusters:
        clusters[dict[key]] = [dict[i] for i in value]

    output_file = os.path.splitext(input_file_name)
    output_file = output_file[0] + "_result.json"
    with open(output_file, "w") as f:
        json.dump(clusters, f, indent=4)

    return clusters

if __name__ == "__main__":
    input_file_name, clusters, delimiter = parse_args(sys.argv[1:])

    if clusters is None:
        clusters = 2

    if delimiter is None:
        delimiter = ";"

    data4, dict = parse_csv(input_file_name, delimiter)
    k_medoids4 = KMedoids(n_cluster=clusters, max_iter=1000)
    k_medoids4.fit(data4)

    clusters = create_output_file(dict, k_medoids4.clusters.items(), input_file_name)

    plot_graphs(data4, k_medoids4)
