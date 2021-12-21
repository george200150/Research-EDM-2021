import numpy as np


# TODO: this is the adaptation of Prec from SACI (with tau, delta and NO w) with tau = 0, values should be similar
tau = 0
k = 1


# def loss_metric(points, labels, matrix_d):  # debugging reasons (must verify matrix_d == distances)
def loss_metric(points, labels):
    size = len(points)
    distances = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            if j > i:
                d = distance(points[i], points[j])
                distances[i][j] = d
                distances[j][i] = d
    distances = np.asarray(distances)

    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_3.pkl", "rb")
    # fd = open("C:/Users/George/PycharmProjects/research_edm/old_saci_studia/matrix_distances_4.pkl", "rb")
    # distances = pickle.load(fd)

    closest_indices = np.argsort(distances, axis=0)
    # assert np.allclose(matrix_d, distances)

    # column having index == 0 represents the distance from self to self (but not always....)
    for indx, self_point in enumerate(closest_indices[0]):
        if self_point != indx:
            closest_indices[0, indx], closest_indices[1, indx] = closest_indices[1, indx], closest_indices[0, indx]  # put self in the right place (col 0)
    # we made it always be distance from self to self.

    knns_of_all_xs = closest_indices[1:k+1]

    acc = 0
    for i in range(size):
        # pos = np.argmin(distances[i])
        pos = knns_of_all_xs[0][i]
        if abs(labels[i] - labels[pos]) <= tau:
            acc += 1.0  # / distances[i][j]
    return acc / size


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)