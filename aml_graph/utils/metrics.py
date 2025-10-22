import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def classification_report(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan'),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'pos_rate': float(y_true.mean()),
    }
