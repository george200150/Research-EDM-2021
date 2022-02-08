import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(gts, preds, classes):
    return confusion_matrix(gts, preds, labels=classes)


def distance(p1, p2):  # n-D euclidean distance
    return np.linalg.norm(p1 - p2)


def get_min_nn(target_grade, nns):
    best_nn = nns[0]  # choose the first available nn
    for nn in nns[1:]:
        if nn[1] <= best_nn[1]:  # [0: label, 1: distance]
            if nn[1] < best_nn[1]:  # found smaller distance => update best-nn
                best_nn = nn
            else:
                if nn[1] == best_nn[1] and target_grade != nn[0]:  # found a difficult student; update
                    best_nn = nn  # always favour increasing the difficulty (grade of student != "g")
                # don't update neighbor students already having final grade "g" (we want only difficult ones -> ~= "g")
        # no closer neighbor found

    # only for debugging reasons (to be noted that a simple sort is not enough to satisfy function requirements)
    # nns = sorted(nns, key=lambda x: (x[1], x[0]))
    # best_nn2 = nns[0]
    # try:
    #     assert best_nn2[0] == best_nn[0] and best_nn2[1] == best_nn[1]
    # except AssertionError:
    #     print(f"Difference noted (not a bug necessarily): target = {target_grade}, sort_nn = {best_nn2[0]},"
    #           f"actual_nn = {best_nn[0]}, different_g = {target_grade != best_nn[0]}, "
    #           f"is_dist_eq = {best_nn2[1] == best_nn[1]}")

    return best_nn


def get_nn(student, students):
    student_final_grade, student_idx, student_features = student
    nns = []
    for label, indx, feat_r in students:
        if student_idx == indx:  # avoid identical students
            continue
        neighbor = [label, distance(student_features, feat_r)]
        nns.append(neighbor)
    clossest_nn = get_min_nn(student_final_grade, nns)
    return clossest_nn[0]  # return the grade of the closest neighbor


def diff(g, labels, features):
    """Defined as the ratio of students from "st" whose nn's label differs from "g".

    @param g: grade - integer
    """
    indices = list([x for x in range(len(features))])  # uniquely identify each student

    students = list(zip(labels, indices, features))
    n = 0
    counter = 0
    for student in students:
        if student[0] != g:
            continue
        if get_nn(student, students) != g:  # #(st_g not g_nn)
            counter += 1  # #st_g
        n += 1
    return counter / n  # ratio of students with grade "g" => #(st_g not g_nn) / #st_g


def sum_diff(st, a):
    suma = 0
    for g in range(4, 11):  # grades are in [4,10]
        suma += diff(g, st, a)
    return suma


def get_quality_metrix(st, a):
    """Inversely proportional to the average of the grades' difficulty.

    @param st: labels set - numpy ndarray of dimension (n,a)
    @param a: feature set - numpy ndarray of dimension (a,)
    """
    # QF(A,St) = 1 − frac(∑(g=4..10)(diff(g,St,A)),7)
    st = [int(x) for x in st]  # integer labels (comparable grades)
    return 1 - sum_diff(st, a) / 7
