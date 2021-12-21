from sklearn.metrics import silhouette_score, rand_score, adjusted_mutual_info_score, mutual_info_score, \
    calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score, homogeneity_score, completeness_score, \
    v_measure_score

import numpy as np

from research_edm.evaluation.Prec_ICVS import loss_metric
from research_edm.evaluation.Prec_SACI import edm_loss

"""
Considered metrics:
    - Silhouette Score
    - Rand Index
    - Adjusted Rand Index
    - Mutual Information
    - Calinski-Harabasz Index
    - Davies-Bouldin Index

    - Fowlkes-Mallows score
    - Homogeneity
    - Completeness
    - V-measure
"""


def silhouette__score(xs, preds):
    """
    HIGHER IS BETTER

    The Silhouette Score and Silhouette Plot are used to measure the separation distance between clusters.
    It displays a measure of how close each point in a cluster is to points in the neighbouring clusters.
    This measure has a range of [-1, 1] and is a great tool to visually inspect the similarities within clusters
        and differences across clusters.

    The higher the Silhouette Coefficients (the closer to +1), the further away the cluster’s samples are from
        the neighbouring clusters samples. A value of 0 indicates that the sample is on or very close to the decision
        boundary between two neighbouring clusters.
    Negative values, instead, indicate that those samples might have been assigned to the wrong cluster.
    Averaging the Silhouette Coefficients, we can get to a global Silhouette Score which can be used to describe
        the entire population’s performance with a single value.
    """
    return silhouette_score(xs, preds, metric='euclidean', sample_size=None, random_state=None)


def rand_index(gts, preds):
    """
    HIGHER IS BETTER

    Another commonly used metric is the Rand Index.
    It computes a similarity measure between two clusters by considering all pairs of samples and counting pairs
        that are assigned in the same or different clusters in the predicted and true clusterings.
    The RI can range from zero to 1, a perfect match.
    The only drawback of Rand Index is that it assumes that we can find the ground-truth clusters labels
        and use them to compare the performance of our model, so it is much less useful than the Silhouette Score
        for pure Unsupervised Learning tasks.
    """
    return rand_score(gts, preds)


def adjusted_rand_index(gts, preds):
    """
    HIGHER IS BETTER

    Rand index adjusted for chance.
    The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples
        and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
    The Adjusted Rand Index, similarly to RI, ranges from zero to one,
        with zero equating to random labelling and one when the clusters are identical.
    """
    return adjusted_mutual_info_score(gts, preds, average_method='arithmetic')


def mutual_information(gts, preds):
    """
    HIGHER IS BETTER

    The Mutual Information is another metric often used in evaluating the performance of Clustering algorithms.
    It is a measure of the similarity between two labels of the same data.
    Similarly to Rand Index, one of the major drawbacks of this metric is requiring to know
        the ground truth labels a priori for the distribution.
    Something which is almost never true in real-life scenarios with Custering.
    """
    return mutual_info_score(gts, preds, contingency=None)


def calinski_harabasz_index(xs, preds):
    """
    HIGHER IS BETTER

    Calinski-Harabasz Index is also known as the Variance Ratio Criterion.
    The score is defined as the ratio between the within-cluster dispersion and the between-cluster dispersion.
    The C-H Index is a great way to evaluate the performance of a Clustering algorithm as it
        does not require information on the ground truth labels.
    The higher the Index, the better the performance.
    """
    return calinski_harabasz_score(xs, preds)


def davies_bouldin_index(xs, preds):
    """
    LOWER IS BETTER

    The Davies-Bouldin Index is defined as the average similarity measure of each cluster with its most similar cluster.
    Similarity is the ratio of within-cluster distances to between-cluster distances.
    In this way, clusters which are farther apart and less dispersed will lead to a better score.
    The minimum score is zero, and differently from most performance metrics,
        the lower values the better clustering performance.
    Similarly to the Silhouette Score, the D-B Index does not require the a-priori knowledge of the ground-truth labels,
        but has a simpler implementation in terms of fomulation than Silhouette Score
    """
    return davies_bouldin_score(xs, preds)


def fowlkes_mallows__score(gts, preds):
    """
    HIGHER IS BETTER

    Measure the similarity of two clusterings of a set of points.
    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of the precision and recall
    FMI = TP / sqrt((TP + FP) * (TP + FN))
    """
    return fowlkes_mallows_score(gts, preds)


def homogeneity(gts, preds):
    """
    HIGHER IS BETTER

    Homogeneity metric of a cluster labeling given a ground truth.
    A clustering result satisfies homogeneity if all of its clusters contain only data points which are members
        of a single class.
    This metric is independent of the absolute values of the labels: a permutation of the class or cluster label
        values won’t change the score value in any way.
    """
    return homogeneity_score(gts, preds)


def completeness(gts, preds):
    """
    HIGHER IS BETTER

    Completeness metric of a cluster labeling given a ground truth.
    A clustering result satisfies completeness if all the data points that are members of a given class
        are elements of the same cluster.
    This metric is independent of the absolute values of the labels: a permutation of the class or cluster
        label values won’t change the score value in any way.
    """
    return completeness_score(gts, preds)


def v_measure(gts, preds):
    """
    HIGHER IS BETTER

    The V-measure is the harmonic mean between homogeneity and completeness
    v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    """
    return v_measure_score(gts, preds)


# Function to find points that are closest to centriods
def closestCentriods(data, ini_cent):
    K = np.shape(ini_cent)[0]

    m = np.shape(data)[0]
    idx = np.zeros((m, 1))

    cent_vals = np.zeros((m, K))
    # Subtract each data row with each centroid value and get the different
    # Find sqaured sum of different of eache each row
    for i in range(K):
        Diff = data - ini_cent[i, :]
        cent_vals[:, i] = np.sum(Diff ** 2, axis=1)

    # Return index of minimum value column wise.
    idx = cent_vals.argmin(axis=1)
    return idx


# Function to find/update centriod, mean of a cluster
def calcCentriods(data, idx, K):
    n = np.shape(data)[1]
    centriods = np.zeros((K, n))
    for i in range(K):
        x_indx = [wx for wx, val in enumerate(idx) if val == i]
        centriods[i, :] = np.mean(data[x_indx, :], axis=0)
        # print('mean:', np.mean(data[x_indx, :], axis= 0))
    return centriods


# Function to find euclidean distance between two points
def findDistance(point1, point2):
    eucDis = 0
    for i in range(len(point1)):
        eucDis = eucDis + (point1[i] - point2[i]) ** 2

    return eucDis ** 0.5


# Function to calcualte Dunn Index
def calcDunnIndex(points, cluster):
    # points -- all data points
    # cluster -- cluster centroids

    numer = float('inf')
    for c in cluster:  # for each cluster
        for t in cluster:  # for each cluster
            # print(t, c)
            if (t == c).all():
                continue  # if same cluster, ignore
            ndis = findDistance(t, c)
            # print('Numerator', numerator, ndis)
            numer = min(numer, ndis)  # find distance between centroids

    denom = 0
    for c in cluster:  # for each cluster
        for p in points:  # for each point
            for t in points:  # for each point
                if (t == p).all():
                    continue  # if same point, ignore
                ddis = findDistance(t, p)
                #    print('Denominator', denominator, ddis)
                denom = max(denom, ddis)

    return numer / denom

# - Clustering quality:
# 	(i) Extrinsic Measures which require ground truth labels. Examples are Adjusted Rand index, Fowlkes-Mallows scores,
# 	Mutual information based scores, Homogeneity, Completeness and V-measure.
# 	(ii) Intrinsic Measures that does not require ground truth labels. Some of the clustering performance measures are
# 	Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index etc.


Prec_SACI = edm_loss
Prec_ICVS = loss_metric