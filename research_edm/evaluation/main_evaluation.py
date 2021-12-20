import os
import xlsxwriter
from tqdm import tqdm

from research_edm.DATA.class_mapping import unmap_category, classes_grades, classes_categories
from research_edm.configs.paths import base_dump_xlxs, mapping_dump_base
from research_edm.evaluation.classification_metrics import get_confusion_matrix
from research_edm.evaluation.clustering_metrics import *
from research_edm.inference.model_instantiation import cls_task
from research_edm.io.pickle_io import get_clustering, get_ready_for_eval, get_labels_mapping


def export_metrics_supervised(ready_for_eval, classes, labels_mapping, result_file):
    r"""
                \   Predicted class
    Actual class|  P     N
    P           | TP	FN
    N           | FP	TN

    sklearn documentation cited:
        Thus in binary classification, the count of true negatives is
        :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
        :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    confusion_matrix (DEFAULT from sci-kit learn):
                    9 8 7 6 5 4 3 2 1
                    predicted
    9|-------------------------------|
    8|  TN  /  /  /  /    /   /   /  |
    7|       *  /  /   /   FP  /  /  |
    6|           *   /   /   /   /   |
    5|               * /   /  /   /  |   actual
    4|                   *   /  /  / |
    3|       FN              *   /   |
    2|                           TP  |
    1|-------------------------------|

    cm' = np.flip(confusion_matrix):
    1|-------------------------------|
    2|  TP                           |
    3| /  /  *             FN        |
    4|  /   /  / *                   |
    5|   /   /   /   *               |  actual
    6|    /   /     /    *           |
    7| /   / FP  /   /   /   *       |
    8|  /   /  /   /  /   /  /  TN   |
    9|-------------------------------|
    predicted
    1 2 3 4 5 6 7 8 9

    np.flip(cm') => axes are swapped now; compatible with report table.

    EXAMPLE:
    Consider a matrix with its superior triangle having values greater than 1, that follow a pattern.

    DEFAULT:
    x =
    1.00000,1.01000,1.02000,1.03000,1.04000,1.05000,1.06000
    0.00000,1.07000,1.08000,1.09000,1.10000,1.11000,1.12000
    0.01000,0.02000,1.13000,1.14000,1.15000,1.16000,1.17000
    0.03000,0.04000,0.05000,1.18000,1.19000,1.20000,1.21000
    0.06000,0.07000,0.08000,0.09000,1.22000,1.23000,1.24000
    0.10000,0.11000,0.12000,0.12000,0.13000,1.25000,1.26000
    0.14000,0.15000,0.16000,0.17000,0.18000,0.19000,1.27000

    np.flip(x) =
    1.27000,0.19000,0.18000,0.17000,0.16000,0.15000,0.14000
    1.26000,1.25000,0.13000,0.12000,0.12000,0.11000,0.10000
    1.24000,1.23000,1.22000,0.09000,0.08000,0.07000,0.06000
    1.21000,1.20000,1.19000,1.18000,0.05000,0.04000,0.03000
    1.17000,1.16000,1.15000,1.14000,1.13000,0.02000,0.01000
    1.12000,1.11000,1.10000,1.09000,1.08000,1.07000,0.00000
    1.06000,1.05000,1.04000,1.03000,1.02000,1.01000,1.00000

    np.transpose(np.flip(x)) =
    1.27000,1.26000,1.24000,1.21000,1.17000,1.12000,1.06000
    0.19000,1.25000,1.23000,1.20000,1.16000,1.11000,1.05000
    0.18000,0.13000,1.22000,1.19000,1.15000,1.10000,1.04000
    0.17000,0.12000,0.09000,1.18000,1.14000,1.09000,1.03000
    0.16000,0.12000,0.08000,0.05000,1.13000,1.08000,1.02000
    0.15000,0.11000,0.07000,0.04000,0.02000,1.07000,1.01000
    0.14000,0.10000,0.06000,0.03000,0.01000,0.00000,1.00000
    """
    # evaluate 10-fold
    sum_conf_matrix = None
    for i in tqdm(range(0, 10), desc="k-fold evaluating..."):
        xs, gts, wrapped_model = ready_for_eval[i]
        preds = wrapped_model.model.predict(xs)

        if wrapped_model.task_type == cls_task:
            gts = labels_mapping.inverse_transform(gts)
            preds = labels_mapping.inverse_transform(preds)
        else:
            if wrapped_model.data_type == "categories":
                preds = list([unmap_category(int(round(x))) for x in preds])
            else:
                preds = list([str((int(round(x)))) for x in preds])

        conf_matrix = get_confusion_matrix(gts, preds, classes)  # TODO: https://en.wikipedia.org/wiki/Confusion_matrix
        if sum_conf_matrix is None:
            sum_conf_matrix = conf_matrix
        else:
            sum_conf_matrix += conf_matrix

    sum_conf_matrix = sum_conf_matrix / 10.0
    sum_conf_matrix = np.flip(sum_conf_matrix)  # sklearn uses other confusion matrix format
    sum_conf_matrix = np.transpose(sum_conf_matrix)  # our table uses other confusion matrix format

    # sum_conf_matrix = np.asarray([  # debugging
    #     [1.00000, 1.01000, 1.02000, 1.03000, 1.04000, 1.05000, 1.06000],
    #     [0.00000, 1.07000, 1.08000, 1.09000, 1.10000, 1.11000, 1.12000],
    #     [0.01000, 0.02000, 1.13000, 1.14000, 1.15000, 1.16000, 1.17000],
    #     [0.03000, 0.04000, 0.05000, 1.18000, 1.19000, 1.20000, 1.21000],
    #     [0.06000, 0.07000, 0.08000, 0.09000, 1.22000, 1.23000, 1.24000],
    #     [0.10000, 0.11000, 0.12000, 0.12000, 0.13000, 1.25000, 1.26000],
    #     [0.14000, 0.15000, 0.16000, 0.17000, 0.18000, 0.19000, 1.27000]
    # ])

    workbook = xlsxwriter.Workbook(result_file)
    worksheet = workbook.add_worksheet()

    worksheet.write("A4", "predicted")
    worksheet.write("C1", "actual")

    worksheet.write("B1", "TP")
    worksheet.write("A3", "TP")

    worksheet.write("A" + str(len(classes) + 2), "TN")
    worksheet.write(chr(ord("A") + len(classes)) + "1", "TN")

    xlsx_column = 'B'
    xlsx_line = 3
    for line in sum_conf_matrix:
        for cell in line:
            worksheet.write(xlsx_column + str(xlsx_line), str(cell))
            xlsx_column = chr(ord(xlsx_column) + 1)
        xlsx_column = 'B'
        xlsx_line += 1

    # result_mat = ""  # debugging
    # for line in sum_conf_matrix:  # works fine
    #     line = ' '.join([str(x) for x in line])
    #     result_mat += line
    #     result_mat += "\n"
    # print(result_mat)

    workbook.close()


def export_metrics_unsupervised(xs, preds, gts, result_file):
    workbook = xlsxwriter.Workbook(result_file)
    worksheet = workbook.add_worksheet()

    worksheet.write("A1", "Silhouette Score")
    worksheet.write("B1", "Rand Index")
    worksheet.write("C1", "Adjusted Rand Index")
    worksheet.write("D1", "Mutual Information")
    worksheet.write("E1", "Davies Bouldin Index")

    worksheet.write("F1", "Fowlkes-Mallows Score")
    worksheet.write("G1", "Homogeneity")
    worksheet.write("H1", "Completeness")
    worksheet.write("I1", "V-measure")
    worksheet.write("J1", "Calinski Harabasz Index")

    worksheet.write("K1", "Prec")

    # TODO: HOW DO YOU GET THE PREDS ???
    worksheet.write("A" + str(2), silhouette__score(xs, gts))
    # worksheet.write("B" + str(2), rand_index(gts, preds))
    # worksheet.write("C" + str(2), adjusted_rand_index(gts, preds))
    # worksheet.write("D" + str(2), mutual_information(gts, preds))
    worksheet.write("E" + str(2), davies_bouldin_index(xs, gts))
    # worksheet.write("F" + str(2), fowlkes_mallows__score(gts, preds))
    # worksheet.write("G" + str(2), homogeneity(gts, preds))
    # worksheet.write("H" + str(2), completeness(gts, preds))
    # worksheet.write("I" + str(2), v_measure(gts, preds))
    worksheet.write("J" + str(2), calinski_harabasz_index(xs, gts))

    dict_labels = convert_label_list_to_dict_mapping(xs, gts)
    prec = Prec(xs, dict_labels)
    worksheet.write("K" + str(2), prec)

    workbook.close()


def main_evaluation(results_paths, learning):
    for path in results_paths:
        pre_name, dset_pkl = path.split(os.path.sep)[-2:]
        dset_name = dset_pkl.split(".")[0]
        data_type = "grades" if "note" in dset_name else "categories"

        if data_type == "grades":
            dump_xlsx_file = os.path.join(base_dump_xlxs, learning, "grades", pre_name, dset_name + ".xlsx")
            classes = classes_grades
        else:
            dump_xlsx_file = os.path.join(base_dump_xlxs, learning, "categories", pre_name, dset_name + ".xlsx")
            classes = classes_categories

        if learning == "supervised":  # principle open-closed not respected below...
            for word in ["_asinh", "_identic", "_log", "_norm", "_mlp", "_nb", "_lr", "_sgdr", "_tr", "_poly"]:
                if word in dset_name:
                    dset_name = "".join(dset_name.split(word))
            lb = get_labels_mapping(os.path.join(mapping_dump_base, data_type, dset_name + ".pkl"))
            ready_for_eval = get_ready_for_eval(path)
            export_metrics_supervised(ready_for_eval, classes, lb, dump_xlsx_file)
        else:
            xs, preds, gts = get_clustering(path)
            export_metrics_unsupervised(xs, preds, gts, dump_xlsx_file)


if __name__ == '__main__':
    base_path = 'C:\\Users\\George\\PycharmProjects\\research_edm\\research_edm\\inference\\trained_classifiers\\'

    # REGRESSION
    # paths = [[os.path.join(base_path, 'grades\\identic\\En_plf_2019-2020_note_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\En_plf_2020-2021_(online)_note_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\plf_2019-2020_note_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\plf_2020-2021_(online)_note_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\En_plf_2019-2020_categorii_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\En_plf_2020-2021_(online)_categorii_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\plf_2019-2020_categorii_norm_identic_poly.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\plf_2020-2021_(online)_categorii_norm_identic_poly.pkl')]]

    # NON-NORMALISED
    paths = [[os.path.join(base_path, 'grades\\identic\\En_plf_2019-2020_note_identic_mlp.pkl')],
             [os.path.join(base_path, 'grades\\identic\\En_plf_2020-2021_(online)_note_identic_mlp.pkl')],
             [os.path.join(base_path, 'grades\\identic\\plf_2019-2020_note_identic_mlp.pkl')],
             [os.path.join(base_path, 'grades\\identic\\plf_2020-2021_(online)_note_identic_mlp.pkl')],
             [os.path.join(base_path, 'categories\\identic\\En_plf_2019-2020_categorii_identic_mlp.pkl')],
             [os.path.join(base_path, 'categories\\identic\\En_plf_2020-2021_(online)_categorii_identic_mlp.pkl')],
             [os.path.join(base_path, 'categories\\identic\\plf_2019-2020_categorii_identic_mlp.pkl')],
             [os.path.join(base_path, 'categories\\identic\\plf_2020-2021_(online)_categorii_identic_mlp.pkl')]]

    for paths_to_models in paths:
        # main_evaluation([os.path.join(base_path, x) for x in pair], "unsupervised")
        main_evaluation(paths_to_models, "supervised")
