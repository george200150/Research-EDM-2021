# def prec(matrix):
#     first_line = matrix[0]  # C6:G6
#     first_val = first_line[0]  # C6
#     return first_val / sum(first_line)  # =C6/sum(C6:G6)
#
#
# def recall(matrix):
#     first_column = matrix[:, 0]  # C6:C10
#     first_val = first_column[0]  # C6
#     return first_val / sum(first_column)  # =C6/sum(C6:C10)
#
#
# def f_measure(prec, recall):
#     return 2 / (1 / prec + 1 / recall)  # =2/(1/C14+1/C15)
#
#
# def weight(matrix):
#     first_column = sum(matrix[:, 0])  # sum(C6:C10)
#     matrix_total = sum([sum(line) for line in matrix])  # sum($C6:$G10)
#     return first_column / matrix_total  # =sum(C6:C10)/sum($C6:$G10)


def get_overall_accuracy(cmsr, cmsc, n):
    tp_cells = []
    for i in range(n):
        tp_cells.append(chr(ord(cmsr) + i) + str(cmsc + i))
    nom = "+".join(tp_cells)

    denom = chr(ord(cmsr)) + str(cmsc) + ":" + chr(ord(cmsr) + n-1) + str(cmsc + n-1)
    return f"=({nom})/sum({denom})"


def get_overall_non_acc_metric(cmsr, omsc, n, offset=0):
    cells = []
    for i in range(n):
        t1 = chr(ord(cmsr) + i) + str(omsc + offset)
        t2 = chr(ord(cmsr) + i) + "$" + str(omsc + 3)  # 3=4-1, where 4 is the number of metrics
        cells.append(t1 + "*" + t2)

    nom = "+".join(cells)
    denom = f"${cmsr}${omsc + 3}:${chr(ord(cmsr) + n-1)}${omsc + 3}"
    return f"=({nom})/sum({denom})"
