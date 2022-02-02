import os
import xlsxwriter
from tqdm import tqdm

from research_edm.DATA.class_mapping import unmap_category, classes_grades, categories_type, \
    get_data_type, grades_type, get_data_type_of_dataset, map_category, classes_categories_7, classes_categories_5, \
    classes_categories_2, post_proc_remap_2, post_proc_remap_5
from research_edm.configs.paths import base_dump_xlxs, mapping_dump_base
from research_edm.evaluation.classification_metrics import get_confusion_matrix
from research_edm.evaluation.clustering_metrics import *
from research_edm.evaluation.xslx_metrics import get_overall_accuracy, get_overall_non_acc_metric
from research_edm.inference.model_instantiation import cls_task
from research_edm.io.pickle_io import get_clustering, get_ready_for_eval, get_labels_mapping


def division_check(nom, denom):
    return "=IF(" + denom + "," + nom + "/" + denom + ",0)"


def compound_check(denom1, denom2):
    return "=IF(" + denom1 + "*" + denom2 + ", 2/(1/" + denom1 + "+1/" + denom2 + "),0)"


def write_weights(cmsc, cmsr, i, n, omsc, worksheet):
    nom = "sum(" + chr(ord(cmsr) + i) + str(cmsc) + ":" + chr(ord(cmsr) + i) + str(cmsc + n - 1) + ")"
    denom = "sum($" + chr(ord(cmsr)) + str(cmsc) + ":$" + chr(ord(cmsr) + n - 1) + str(cmsc + n - 1) + ")"
    worksheet.write(chr(ord(cmsr) + i) + str(omsc + 4), division_check(nom, denom))
    # =sum(C6: C10) / sum($C6:$G10)


def write_overall_recall(cmsr, i, omsc, worksheet):
    denom_1 = chr(ord(cmsr) + i) + str(omsc + 1)
    denom_2 = chr(ord(cmsr) + i) + str(omsc + 2)
    worksheet.write(chr(ord(cmsr) + i) + str(omsc + 3), compound_check(denom_1, denom_2))
    # =2 / (1 / C14 + 1 / C15)


def write_overall_prec(cmsc, cmsr, i, n, omsc, worksheet):
    nom = chr(ord(cmsr) + i) + str(cmsc + i)
    denom = "sum(" + chr(ord(cmsr) + i) + str(cmsc) + ":" + chr(ord(cmsr) + i) + str(cmsc + n - 1) + ")"
    worksheet.write(chr(ord(cmsr) + i) + str(omsc + 2), division_check(nom, denom))
    # =C6 / sum(C6: C10)


def write_overall_accuracy(cmsc, cmsr, i, n, omsc, worksheet):
    # cannot use chr(ord("9") + 1), because ASCII table does not have a "10"
    nom = chr(ord(cmsr) + i) + str(cmsc + i)
    denom = "sum(" + chr(ord(cmsr)) + str(cmsc + i) + ":" + chr(ord(cmsr) + n - 1) + str(cmsc + i) + ")"
    worksheet.write(chr(ord(cmsr) + i) + str(omsc + 1), division_check(nom, denom))
    # =C6/sum(C6:G6)


def export_metrics_supervised(no_classes, ready_for_eval, classes, labels_mapping, result_file):
    if len(classes) == 7 or no_classes == 7:  # grades
        classes_categories = classes_categories_7
    elif no_classes == 5:  # E V G S F
        classes_categories = classes_categories_5
    elif no_classes == 2:  # P F
        classes_categories = classes_categories_2
    else:
        raise ValueError("No such class mapping!")

    # evaluate 10-fold
    sum_conf_matrix = None
    for i in tqdm(range(0, 10), desc="k-fold evaluating..."):
        xs, gts, wrapped_model = ready_for_eval[i]
        preds = wrapped_model.model.predict(xs)

        if wrapped_model.task_type == cls_task:
            gts = labels_mapping.inverse_transform(gts)
            preds = labels_mapping.inverse_transform(preds)  # TODO: check type
            # post-process predictions
            if len(classes) == 7:
                pass
            elif no_classes == 5:
                preds = np.asarray([post_proc_remap_5[x] for x in preds])
                gts = np.asarray([post_proc_remap_5[x] for x in gts])
            elif no_classes == 2:
                preds = np.asarray([post_proc_remap_2[x] for x in preds])
                gts = np.asarray([post_proc_remap_2[x] for x in gts])
            else:
                raise ValueError("Cannot remap predictions!")
        else:
            if wrapped_model.data_type == categories_type:
                preds = list([unmap_category(no_classes, int(round(x))) for x in preds])
            else:
                preds = list([str((int(round(x)))) for x in preds])

        conf_matrix = get_confusion_matrix(gts, preds, classes)  # TODO: still, please verify label order integrity...
        if sum_conf_matrix is None:
            sum_conf_matrix = conf_matrix
        else:
            sum_conf_matrix += conf_matrix

    # sum_conf_matrix = sum_conf_matrix / 10.0
    sum_conf_matrix = np.flip(sum_conf_matrix)  # sklearn uses other confusion matrix format
    sum_conf_matrix = np.transpose(sum_conf_matrix)  # our table uses other confusion matrix format

    workbook = xlsxwriter.Workbook(result_file)
    worksheet = workbook.add_worksheet()

    worksheet.write("A4", "Predict")
    worksheet.write("A5", "ed class")

    worksheet.write("E1", "Actual class")

    conf_matrix_starting_row, conf_matrix_starting_column = "C", 3
    cmsc = conf_matrix_starting_column
    cmsr = conf_matrix_starting_row

    for idx, categ in enumerate(classes_categories):
        worksheet.write(chr(ord(cmsr) + idx) + str(cmsc-1), categ)
        worksheet.write(chr(ord(cmsr)-1) + str(cmsc + idx), categ)

    xlsx_column = cmsr
    xlsx_line = cmsc
    for line in sum_conf_matrix:
        for cell in line:
            worksheet.write_number(xlsx_column + str(xlsx_line), cell)
            xlsx_column = chr(ord(xlsx_column) + 1)
        xlsx_column = cmsr
        xlsx_line += 1

    ####################################################################################################################

    n = len(classes_categories)

    overall_metrics_starting_row, overall_metrics_starting_column = "J", 13
    omsc = overall_metrics_starting_column
    omsr = overall_metrics_starting_row

    worksheet.write("A" + str(omsc), "Acc")
    worksheet.write("A" + str(omsc+1), "Prec")
    worksheet.write("A" + str(omsc+2), "Recall")
    worksheet.write("A" + str(omsc+3), "F-measure")
    worksheet.write("A" + str(omsc+4), "Weight")

    worksheet.write(chr(ord(omsr)) + str(omsc-1), "Overall/weigthed")
    worksheet.write(chr(ord(omsr)) + str(omsc), get_overall_accuracy(cmsr, cmsc, n))

    for i in range(0, n):
        write_overall_accuracy(cmsc, cmsr, i, n, omsc, worksheet)
        write_overall_prec(cmsc, cmsr, i, n, omsc, worksheet)
        write_overall_recall(cmsr, i, omsc, worksheet)
        write_weights(cmsc, cmsr, i, n, omsc, worksheet)

        worksheet.write(chr(ord(omsr)) + str(omsc+1), get_overall_non_acc_metric(cmsr, omsc+1, n))
        worksheet.write(chr(ord(omsr)) + str(omsc+2), get_overall_non_acc_metric(cmsr, omsc+1, n, offset=1))
        worksheet.write(chr(ord(omsr)) + str(omsc+3), get_overall_non_acc_metric(cmsr, omsc+1, n, offset=2))

    ####################################################################################################################

    workbook.close()


def export_metrics_unsupervised(no_classes, xs, preds, gts, result_file):
    # convert labels to integers (order relationship is better defined)
    if get_data_type_of_dataset(no_classes, gts) == grades_type:
        gts = list([int(y) for y in gts])
    else:
        gts = list([map_category(no_classes, y) for y in gts])

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

    worksheet.write("K1", "Prec SACI")
    worksheet.write("L1", "Prec ICVS")

    tau = 1  # 0, 1
    k = 3  # 1, 3, 5

    # this row in the table uses the gts instead of the preds
    worksheet.write("A" + str(2), silhouette__score(xs, gts))
    # 1 always
    # 1 always
    worksheet.write("D" + str(2), mutual_information(gts, gts))
    worksheet.write("E" + str(2), davies_bouldin_index(xs, gts))
    # 1 always
    # 1 always
    # 1 always
    # 1 always
    worksheet.write("J" + str(2), calinski_harabasz_index(xs, gts))
    worksheet.write("K" + str(2), Prec_SACI(xs, gts, tau, k))
    worksheet.write("L" + str(2), Prec_ICVS(xs, gts))
    worksheet.write("N" + str(2), "Metrics using ground truths instead of preds")

    # this row in the table uses the parameters the formulas expect
    worksheet.write("A" + str(3), silhouette__score(xs, preds))
    worksheet.write("B" + str(3), rand_index(gts, preds))
    worksheet.write("C" + str(3), adjusted_rand_index(gts, preds))
    worksheet.write("D" + str(3), mutual_information(gts, preds))
    worksheet.write("E" + str(3), davies_bouldin_index(xs, preds))
    worksheet.write("F" + str(3), fowlkes_mallows__score(gts, preds))
    worksheet.write("G" + str(3), homogeneity(gts, preds))
    worksheet.write("H" + str(3), completeness(gts, preds))
    worksheet.write("I" + str(3), v_measure(gts, preds))
    worksheet.write("J" + str(3), calinski_harabasz_index(xs, preds))
    worksheet.write("K" + str(3), Prec_SACI(xs, preds, tau, k))
    worksheet.write("L" + str(3), Prec_ICVS(xs, preds))
    worksheet.write("N" + str(3), "Real value of metrics")

    workbook.close()


def main_evaluation(no_classes, results_paths, learning):
    for path in results_paths:
        pre_name, dset_pkl = path.split(os.path.sep)[-2:]
        dset_name = dset_pkl.split(".")[0]
        data_type = get_data_type(dset_name)

        if data_type == grades_type:
            dump_xlsx_file = os.path.join(base_dump_xlxs, learning, grades_type, pre_name, dset_name + str(no_classes) + ".xlsx")
            classes = classes_grades
        else:
            dump_xlsx_file = os.path.join(base_dump_xlxs, learning, categories_type, pre_name, dset_name + str(no_classes) + ".xlsx")
            if no_classes == 2:
                classes = classes_categories_2
            elif no_classes == 5:
                classes = classes_categories_5
            elif no_classes == 7:
                classes = classes_categories_7
            else:
                raise ValueError("No such class mapping!")

        if learning == "supervised":  # principle open-closed not respected below...
            for word in ["_asinh", "_identic", "_log", "_norm", "_mlp", "_nb", "_lr", "_sgdr", "_tr", "_poly"]:
                if word in dset_name:
                    dset_name = "".join(dset_name.split(word))
            lb = get_labels_mapping(os.path.join(mapping_dump_base, data_type, dset_name + ".pkl"))
            ready_for_eval = get_ready_for_eval(path)
            export_metrics_supervised(no_classes, ready_for_eval, classes, lb, dump_xlsx_file)
        else:
            xs, preds, gts = get_clustering(path)
            export_metrics_unsupervised(no_classes, xs, preds, gts, dump_xlsx_file)


if __name__ == '__main__':
    # base_path = 'C:\\Users\\George\\PycharmProjects\\Research-EDM-2021\\research_edm\\inference\\trained_classifiers\\'
    base_path = 'C:\\Users\\George\\PycharmProjects\\Research-EDM-2021\\research_edm\\clustering\\clustering_dump\\'

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
    # paths = [[os.path.join(base_path, 'grades\\identic\\En_plf_2019-2020_note_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\En_plf_2020-2021_(online)_note_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\plf_2019-2020_note_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'grades\\identic\\plf_2020-2021_(online)_note_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\En_plf_2019-2020_categorii_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\En_plf_2020-2021_(online)_categorii_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\plf_2019-2020_categorii_identic_mlp.pkl')],
    #          [os.path.join(base_path, 'categories\\identic\\plf_2020-2021_(online)_categorii_identic_mlp.pkl')]]

    paths = [[os.path.join(base_path, 'identic\\En_plf_2019-2020_note_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\En_plf_2020-2021_(online)_note_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\plf_2019-2020_note_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\plf_2020-2021_(online)_note_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\En_plf_2019-2020_categorii_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\En_plf_2020-2021_(online)_categorii_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\plf_2019-2020_categorii_norm_identic_umap.pkl')],
             [os.path.join(base_path, 'identic\\plf_2020-2021_(online)_categorii_norm_identic_umap.pkl')]]

    for paths_to_models in paths:
        main_evaluation(2, paths_to_models, "unsupervised")
        # main_evaluation(5, paths_to_models, "unsupervised")
        # main_evaluation(7, paths_to_models, "unsupervised")
        # main_evaluation(7, paths_to_models, "supervised")
