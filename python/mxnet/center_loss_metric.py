
"""
Center loss metric:
Apply MSE metric
Only use distance between examples and it's class center vector
"""

from .metric import EvalMetric, check_label_shapes
from . import ndarray
import numpy


class CenterLossMSE(EvalMetric):
    """ return MSE metric on examples and it's class center vector """
    def __init__(self, **kwargs):
        super(CenterLossMSE, self).__init__('center_loss_mse')
        try:
            self.label_array_idx = kwargs['label_array_idx']
        except KeyError:
            self.label_array_idx = 1

    def update(self, labels, preds):
        """ preds format: dimension (M, ), M examples, preds[i] = l2_norm(feat_i - center_vec(label[i])) """
        check_label_shapes(labels, preds)

        label = labels[self.label_array_idx].asnumpy().astype('int32')
        dis_label = preds[self.label_array_idx].asnumpy()

        check_label_shapes(label, dis_label)
        self.sum_metric += numpy.sum(dis_label) / 0.5
        #self.sum_metric += numpy.sum(dis_label * numpy.sign(label)) / 0.5
        self.num_inst += label.shape[0]


class CenterLossAccuracy(EvalMetric):
    """Calculate accuracy"""

    def __init__(self, **kwargs):
        super(CenterLossAccuracy, self).__init__('center_loss_accuracy')
        try:
            self.label_array_idx = kwargs['label_array_idx']
        except KeyError:
            self.label_array_idx = 0

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        pred_label = ndarray.argmax_channel(preds[self.label_array_idx]).asnumpy().astype('int32')
        label = labels[self.label_array_idx].asnumpy().astype('int32')

        check_label_shapes(label, pred_label)

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

