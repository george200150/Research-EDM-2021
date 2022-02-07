import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(gts, preds, classes):
    return confusion_matrix(gts, preds, labels=classes)


def distance(p1, p2):  # n-D euclidean distance
    return np.linalg.norm(p1 - p2)


def get_min_nn(target_grade, nns):
    min_stud = None
    min_stud_non_target = None
    for nn in nns:
        if min_stud is None or nn[1] < min_stud[1]:  # [0: label, 1: distance]
            if nn[0] == target_grade:
                min_stud = nn
            else:
                min_stud_non_target = nn
    if min_stud_non_target is None:
        return min_stud
    if min_stud is None:
        return min_stud_non_target
    if min_stud_non_target[1] <= min_stud[1]:  # always favour increasing the difficulty (grade of student != "g")
        return min_stud_non_target
    return min_stud


def get_nn(feat, labels, features):
    nns = []
    for lab, feat_r in zip(labels, features):
        if feat[0] == feat_r[0]:  # avoid identical students
            continue
        nns.append([lab, distance(feat[1:], feat_r[1:])])
    clossest_nn = get_min_nn(feat[0], nns)
    return clossest_nn[0]  # return the grade of the closest neighbour


def diff(g, labels, features):
    """Defined as the ratio of students from "st" whose nn's label differs from "g".

    @param g: grade - integer
    """
    indx = np.expand_dims(np.asarray([x for x in range(len(features))]), 1)  # uniquely identify each student
    features = np.concatenate((indx, features), axis=1)

    n = 0
    counter = 0
    for result, grades in zip(labels, features):
        if int(result) != g:
            continue
        if int(get_nn(grades, labels, features)) != g:  # #(st_g not g_nn)
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

    @param st: labels set - numpy ndarray of dimension (n,a,1)  # TODO: check dims
    @param a: feature set - numpy ndarray of dimension (a,1)
    """
    # QF(A,St) = 1 − frac(∑(g=4..10)(diff(g,St,A)),7)
    return 1 - sum_diff(st, a) / 7
