import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

def standartify_clusters(clusters):
    standartified = clusters.copy()
    for i, val in enumerate(np.unique(clusters)):
        standartified[clusters == val] = i
    return standartified

def nmi_geom(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, average_method='geometric')

def cost_matrix(target, pred):
    target_vals = sorted(list(set(target)))
    pred_vals = sorted(list(set(pred)))
    C = np.zeros((len(pred_vals), len(target_vals)))
    for row in pred_vals:
        for col in target_vals:
            C[row, col] = ((pred == row) * (target == col)).sum()
    return C

def accuracy_with_reassignment(y_true, y_pred):
    C = cost_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    return C[row_ind, col_ind].sum() / len(y_true)