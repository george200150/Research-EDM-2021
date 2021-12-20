from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

from read_data import read_students
from research_edm.evaluation.Prec3_OG import loss_metric

path = r"C:\Users\George\PycharmProjects\research_edm\old_saci_studia"

# Import dataset
entire_dataset, entire_dataset_final_grade = read_students(path + '\\SDA_905instances_3Features(fara a3).txt', 'regression', 3)
# entire_dataset, entire_dataset_final_grade = read_students(path + '\\SDA_905instances_4Features(+1 output grade).txt', 'regression', 4)

# Pentu cand doar setul de training si cel de testing sunt disponibile si trebuie recreat intregul set de date
# Import training dataset
# train_x, train_y = read_students(path + '\\SDA_905instances_4Features(+1 output grade).txt', 'regression', 4)

# Import testing dataset
# test_x, test_y = read_students(path + '\\test2.txt', 'regression')

# entire_dataset = np.concatenate((train_x, test_x))
# entire_dataset_final_grade = np.concatenate((train_y, test_y))

# pca = PCA(n_components=2)
# pca = UMAP(n_components=2, n_neighbors=30)
pca = UMAP(n_components=2, n_neighbors=15)
# pca = TSNE(n_components=2)

entire_dataset = np.asarray(entire_dataset)

mean = np.mean(entire_dataset, axis=0)
stdev = np.std(entire_dataset, axis=0)
print(f"MEAN: {mean}, STDEV: {stdev}")

entire_dataset = (entire_dataset - mean) / stdev

principalComponents = pca.fit_transform(entire_dataset)

# LOSS

# A value of 0 for τ means that x and pair are considered correctly
# mapped only if a = b and this leads to a maximum accuracy
# of the computations. Larger values for τ (e.g. 1) weaken the
# constraint in computing the Prec value, i.e. x and pair are
# considered correctly mapped if |a −b| ≤ τ.
#tau = 0  # 1


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


def edm_loss(xs, ys):
    tau = 1  # 0, 1
    k = 3  # 1, 3, 5
    data = np.array([[x[0][0], x[0][1], x[1]] for x in list(zip(xs, ys))])  # nd_array of (x1, x2, y)

    from old_saci_studia.some_tries import knn_total, knn_by_point
    # neighk_x_matrix, matrix_dist = knn_total(data)  # debug reasons
    neighk_x_matrix = knn_total(data)  # adiacence matrix

    loss = 0

    for indx, pair in enumerate(data):  # for each y in D (as formula says)
        indx = knn_by_point(neighk_x_matrix, k, indx)
        loss += prec(pair, data[indx], k, tau)

    # return loss / len(data), matrix_dist  # debug reasons
    return loss / len(data)


# cost, matrix_d = edm_loss(principalComponents, entire_dataset_final_grade)  # debug reasons
cost1 = edm_loss(principalComponents, entire_dataset_final_grade)
print(cost1)
cost = edm_loss(entire_dataset, entire_dataset_final_grade)
print(cost)
exit(0)


# acc = loss_metric(principalComponents, entire_dataset_final_grade, matrix_d)  # debug reasons
acc = loss_metric(principalComponents, entire_dataset_final_grade)
print(acc)

exit(0)

# LOSS

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(' ', fontsize=15)
ax.set_ylabel(' ', fontsize=15)
ax.set_title('2 component UMAP n_neighbors=15', fontsize=20)

# indicesToKeep = entire_dataset_final_grade >= 5
# print(sum(indicesToKeep))
# ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="b")
#
# indicesToKeep2 = entire_dataset_final_grade < 5
# print(sum(indicesToKeep2))
# ax.scatter(principalComponents[indicesToKeep2, 0], principalComponents[indicesToKeep2, 1], color="r")

mark = 'o'
indicesToKeep = entire_dataset_final_grade == 4
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="blue", marker=mark)

indicesToKeep = entire_dataset_final_grade == 5
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="green", marker=mark)

indicesToKeep = entire_dataset_final_grade == 6
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="pink", marker=mark)

indicesToKeep = entire_dataset_final_grade == 7
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="cyan", marker=mark)

indicesToKeep = entire_dataset_final_grade == 8
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="yellow", marker=mark)

indicesToKeep = entire_dataset_final_grade == 9
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="magenta", marker=mark)

indicesToKeep = entire_dataset_final_grade == 10
print(sum(indicesToKeep))
ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], color="red", marker=mark)

# for i in range(principalComponents.shape[0]):
#     if entire_dataset_final_grade[i]==4:
#         ax.annotate("4", (principalComponents[i, 0], principalComponents[i, 1]), color="blue")
#     if entire_dataset_final_grade[i]==5:
#         ax.annotate("5", (principalComponents[i, 0], principalComponents[i, 1]), color="green")
#     if entire_dataset_final_grade[i]==6:
#         ax.annotate("6", (principalComponents[i, 0], principalComponents[i, 1]), color="pink")
#     if entire_dataset_final_grade[i]==7:
#         ax.annotate("7", (principalComponents[i, 0], principalComponents[i, 1]), color="cyan")
#     if entire_dataset_final_grade[i]==8:
#         ax.annotate("8", (principalComponents[i, 0], principalComponents[i, 1]), color="yellow")
#     if entire_dataset_final_grade[i]==9:
#         ax.annotate("9", (principalComponents[i, 0], principalComponents[i, 1]), color="magenta")
#     if entire_dataset_final_grade[i]==10:
#         ax.annotate("10", (principalComponents[i, 0], principalComponents[i, 1]), color="red")

# for i in range(principalComponents.shape[0]):
#     if entire_dataset_final_grade[i]==4:
#         ax.annotate("4", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==5:
#         ax.annotate("5", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==6:
#         ax.annotate("6", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==7:
#         ax.annotate("7", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==8:
#         ax.annotate("8", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==9:
#         ax.annotate("9", (principalComponents[i, 0], principalComponents[i, 1]), color="black")
#     if entire_dataset_final_grade[i]==10:
#         ax.annotate("10", (principalComponents[i, 0], principalComponents[i, 1]), color="black")


ax.legend(['4', '5', '6', '7', '8', '9', '10'])
ax.grid()
plt.show()
