import numpy as np
from scipy.spatial import distance


# N = 10  # The number of points
# points = np.random.rand(N, 3)
# print("points = \n", points)
# print("================================================")
#
# D = distance.squareform(distance.pdist(points))
# print("round = \n", np.round(D, 1))  # Rounding to fit the array on screen
# print("================================================")
#
# closest = np.argsort(D, axis=1)
# print("closest = \n", closest)
# print("================================================")
#
# k = 3  # For each point, find the 3 closest points
# print(closest[:, 1:k+1])


def knn_total1(pairs, k):
    points = np.asarray([data[:-1] for data in pairs]).squeeze()
    D = distance.squareform(distance.pdist(points))
    closest_indices = np.argsort(D, axis=1)
    # return closest_indices[:, 1:k+1]
    return closest_indices[:, 1:k+1]


def knn_by_point1(knn_matrix, indices):
    return knn_matrix[indices]


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def knn_total(pairs):
    n = len(pairs)
    matrix_distances = np.zeros(shape=(n, n))
    for i, pair1 in enumerate(pairs):
        for j, pair2 in enumerate(pairs):
            if j > i:
                d = euclidean_distance(pair1[:-1], pair2[:-1])
                matrix_distances[i][j] = d
                matrix_distances[j][i] = d

    # import pickle
    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_3.pkl", "wb")
    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_4.pkl", "wb")
    # pickle.dump(matrix_distances, fd)

    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_3.pkl", "rb")
    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_4.pkl", "rb")
    # matrix_distances = pickle.load(fd)

    closest_indices = np.argsort(matrix_distances, axis=1)

    # return closest_indices, matrix_distances  # debug reasons
    return closest_indices


def knn_by_point(knn_matrix, k, n_th_x):
    # column having index == 0 represents the distance from self to self (but not always....)
    for indx, self_point in enumerate(knn_matrix[:, 0]):
        if self_point != indx:
            knn_matrix[indx, 0], knn_matrix[indx, 1] = knn_matrix[indx, 1], knn_matrix[indx, 0]  # put self in the right place (col 0)
    # we made it always be distance from self to self.
    knns_of_all_xs = knn_matrix[:, 1:k+1]
    return knns_of_all_xs[n_th_x]
