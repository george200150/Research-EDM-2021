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
