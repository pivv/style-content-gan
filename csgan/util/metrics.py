import sys
import os

import numpy as np

import sklearn
from collections import OrderedDict
import sklearn.metrics

import matplotlib.pyplot as plt


class Metrics(object):
    def __init__(self):
        super().__init__()
        self._odict: OrderedDict = OrderedDict()
        self.keys = ('confusion_matrix', 'roc_auc', 'f1_score',
                     'accuracy', 'balanced_accuracy', 'recall', 'precision',
                     'roc_curve')
        for key in self.keys:
            self._odict[key] = None

    def __getitem__(self, item: str):
        return self._odict[item]

    def plot(self, ax: plt.Axes = None) -> None:
        for key in self.keys:
            if self._odict[key] is None:
                continue
            elif key == 'roc_curve':
                fpr, tpr, thresholds = self._odict[key]

                if ax is None:
                    fig = plt.figure(figsize=(6, 6))
                    ax = plt.subplot(1, 1, 1)
                ax.plot(fpr, tpr)
                ax.set_title("ROC Curve")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.axis('equal')
                if ax is None:
                    plt.show()

    def __str__(self) -> str:
        s = ''
        for key in self.keys:
            if self._odict[key] is None:
                continue
            elif key == 'roc_curve':
                continue
            elif key == 'confusion_matrix':
                m00, m01, m10, m11 = (self._odict[key][0, 0], self._odict[key][0, 1],
                                      self._odict[key][1, 0], self._odict[key][1, 1])
                s += f'{key}: [[{m00}, {m01}], [{m10}, {m11}]]\n'
            else:
                s += f'{key}: {self._odict[key]:.5f}\n'
        return s

    def compute(self, pred: np.ndarray, gold: np.ndarray, signals: np.ndarray = None, topk: int = None) -> OrderedDict:
        if topk is not None:
            assert(signals is not None)
            indices = np.concatenate([np.argsort(signals)[:topk], np.argsort(signals)[-topk:]])
            pred, gold, signals = pred[indices], gold[indices], signals[indices]
        sign_pred = pred.astype('int')
        sign_gold = gold.astype('int')
        self._odict['confusion_matrix'] = sklearn.metrics.confusion_matrix(sign_gold, sign_pred)
        self._odict['roc_auc'] = sklearn.metrics.roc_auc_score(sign_gold, signals) if signals is not None else None
        self._odict['f1_score'] = sklearn.metrics.f1_score(sign_gold, sign_pred)
        self._odict['accuracy'] = sklearn.metrics.accuracy_score(sign_gold, sign_pred)
        self._odict['balanced_accuracy'] = sklearn.metrics.balanced_accuracy_score(sign_gold, sign_pred)
        self._odict['recall'] = sklearn.metrics.recall_score(sign_gold, sign_pred)
        self._odict['precision'] = sklearn.metrics.precision_score(sign_gold, sign_pred)
        self._odict['roc_curve'] = sklearn.metrics.roc_curve(sign_gold, signals) if signals is not None else None
        return self._odict
