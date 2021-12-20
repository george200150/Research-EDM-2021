import numpy as np
from research_edm.evaluation.classification_metrics import get_confusion_matrix

classes = ['4', '5', '6', '7', '8', '9', '10']


def test():
    gts =   ['6', '7',  '9', '4', '7', '7', '6', '5', '8', '7', '9', '10', '9', '8', '6', '10', '7']
    preds = ['6', '7', '10', '5', '7', '7', '6', '5', '8', '7', '9', '10', '9', '8', '6', '10', '7']

    # Am luat si le-am numarat. Da, este bine.
    #                       \         Actual class top      |
    #                        \   4   5   6   7   8   9  10  |
    #                        4   0   0   0   0   0   0   0  |
    #                        5   1   1   0   0   0   0   0  |
    #                        6   0   0   3   0   0   0   0  |
    # Predicted class left   7   0   0   0   5   0   0   0  |
    #                        8   0   0   0   0   2   0   0  |
    #                        9   0   0   0   0   0   2   0  |
    #                       10   0   0   0   0   0   1   2  |
    #

    #                       \         Actual class top      |
    #                        \  10   9   8   7   6   5   4  |
    #                       10   2   1   0   0   0   0   0  |
    #                        9   0   2   0   0   0   0   0  |
    #                        8   0   0   2   0   0   0   0  |
    # Predicted class left   7   0   0   0   5   0   0   0  |
    #                        6   0   0   0   0   3   0   0  |
    #                        5   0   0   0   0   0   1   1  |
    #                        4   0   0   0   0   0   0   0  |

    conf_matrix_gt = np.asarray(
        [[2, 1, 0, 0, 0, 0, 0],
         [0, 2, 0, 0, 0, 0, 0],
         [0, 0, 2, 0, 0, 0, 0],
         [0, 0, 0, 5, 0, 0, 0],
         [0, 0, 0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0]
         ])

    conf_matrix = get_confusion_matrix(gts, preds, classes)

    conf_matrix = np.transpose(conf_matrix)  # our table uses other confusion matrix format

    assert np.allclose(conf_matrix, conf_matrix_gt)
    print("tests passed")


if __name__ == '__main__':
    test()
