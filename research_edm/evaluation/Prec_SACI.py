import numpy as np


def delta(a, b, tau):
    return 1 if abs(a - b) <= tau else 0


def w(a, b):
    return abs(a - b) / 10


def label(pair):  # label(y) == f(y) its final examination
    return pair[-1]


def prec(pair, neighk_x, k, tau):
    s_nom = 0
    s_denom = k  # nominator / k +
    for x in neighk_x:  # for each x of Nk (as formula says)
        s_nom += delta(label(x), label(pair), tau)  # TODO: label(x) - there is no label
        s_denom += w(label(x), label(pair))
    return s_nom / s_denom


def edm_loss(xs, ys, tau, k):
    # from research_21_22.experiments.evaluation_of_clustering.Prec_SACI import knn_total, knn_by_point
    data = np.array([[x[0][0], x[0][1], x[1]] for x in list(zip(xs, ys))])  # nd_array of (x1, x2, y)

    # neighk_x_matrix, matrix_dist = knn_total(data)  # debug reasons
    neighk_x_matrix = knn_total(data)  # adiacence matrix

    loss = 0

    for indx, pair in enumerate(data):  # for each y in D (as formula says)
        indx = knn_by_point(neighk_x_matrix, k, indx)
        loss += prec(pair, data[indx], k, tau)

    # return loss / len(data), matrix_dist  # debug reasons
    return loss / len(data)


# //////////////////////////////// different fule in the original loss implementation ////////////////////////////////

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
