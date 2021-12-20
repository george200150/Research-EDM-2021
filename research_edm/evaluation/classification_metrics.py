from sklearn.metrics import confusion_matrix


def get_confusion_matrix(gts, preds, classes):
    return confusion_matrix(gts, preds, labels=classes)
