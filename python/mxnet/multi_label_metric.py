
"""
Metrics for Multi-label Learning
Support Metrics:
    - Example based:
        - subset accuracy
        - hamming loss
        - accuracy, precision, recall, f1-score
    - Label based:
        - macro/micro metrics
        - accuracy, precision, recall, f1-score
    - Ranking Loss
"""


from .metric import EvalMetric, check_label_shapes
import numpy

def prob2label(pred_prob, thresh=0.5):
    """ will return binary labels (1 for positive, 0 for negative), positive if prob >= thresh """
    label = numpy.zeros(pred_prob.shape, dtype=numpy.int32)
    pos = numpy.where(pred_prob >= thresh)
    label[pos] = 1
    return label


class SubsetAccuracy(EvalMetric):
    """ output the fraction of totally correct instance, i.e. pred(x) = label(x) """
    def __init__(self, threshold=0.5):
        super(SubsetAccuracy, self).__init__('subset_accuracy')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)
            for pred, l in zip(pred_label, label):
                real = numpy.where(l >=0)
                if numpy.all(pred[real] == l[real]):
                    self.sum_metric += 1

            self.num_inst += label.shape[0]

class HammingLoss(EvalMetric):
    """ output the fraction of mis-classified labels, i.e. |pred(x) != label(x)| / |label(x)|, |*| denotes the number of elements in set """
    def __init__(self, threshold=0.5):
        super(HammingLoss, self).__init__('hamming_loss')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)
            real_label_idx = numpy.where(label >= 0)
            self.sum_metric += len(numpy.where(pred_label[real_label_idx] != label[real_label_idx])[0]) / float(label.shape[1])

            for pred, l in zip(pred_label, label):
                real = numpy.where(l >= 0)
                self.sum_metric += len(numpy.where(pred[real] != l[real])[0]) / float(l[real].shape[0])

            self.num_inst += label.shape[0]

class AccuracyExam(EvalMetric):
    """ output the instance-level accuracy, i.e. |pred(x) ^ label(x)|/|pred(x) u label(x)|, |*| denotes the number of element in set """
    def __init__(self, threshold=0.5):
        super(AccuracyExam, self).__init__('accuracy_exam')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)

            for pred, l in zip(pred_label, label):
                real = numpy.where(l >= 0)
                num_pred_and_l = len(numpy.where(numpy.logical_and(pred == 1, l == 1))[0])
                num_pred_or_l = len(numpy.where(numpy.logical_or(pred[real] == 1, l[real] == 1))[0])
                self.sum_metric += num_pred_and_l / float(num_pred_or_l) if num_pred_or_l > 0 else 0

            self.num_inst += label.shape[0]

class PrecisionExam(EvalMetric):
    """ output the instance-level precision, i.e. |pred(x) ^ label(x)|/|pred(x)|, |*| denotes the number of element in set """
    def __init__(self, threshold=0.5):
        super(AccuracyExam, self).__init__('precision_exam')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)

            for pred, l in zip(pred_label, label):
                real = numpy.where(l >= 0)
                num_pred_and_l = len(numpy.where(numpy.logical_and(pred == 1, l == 1))[0])
                num_pred = len(numpy.where(pred[real] == 1)[0])
                self.sum_metric += num_pred_and_l / float(num_pred) if num_pred > 0 else 0

            self.num_inst += label.shape[0]

class RecallExam(EvalMetric):
    """ output the instance-level recall, i.e. |pred(x) ^ label(x)|/|label(x)|, |*| denotes the number of element in set """
    def __init__(self, threshold=0.5):
        super(AccuracyExam, self).__init__('recall_exam')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)

            for pred, l in zip(pred_label, label):
                num_pred_and_l = len(numpy.where(numpy.logical_and(pred == 1, l == 1))[0])
                num_l = len(numpy.where(l == 1)[0])
                self.sum_metric += num_pred_and_l / float(num_l) if num_l > 0 else 0

            self.num_inst += label.shape[0]

class F1Exam(EvalMetric):
    """ output instance-level f1 score """
    def __init__(self, threshold=0.5):
        super(F1Exam, self).__init__('f1_exam')
        self.recall = RecallExam(threshold=threshold)
        self.precision = PrecisionExam(threshold=threshold)

    def reset(self):
        self.recall.reset()
        self.precision.reset()

    def update(self, labels, preds):
        self.recall.update(labels, preds)
        self.precision.update(labels, preds)

    def get(self):
        recall = self.recall.get()
        precision = self.precision.get()
        if recall + precision > 0:
            return (self.name, (recall * precision) / (recall + precision))
        else:
            return (self.name, 0.0)

def get_diff_pairs(label):
    """ return all possible pairs of label index (p_idx, n_idx) of an instance """
    pos = numpy.where(label == 1)
    neg = numpy.where(label == 0)
    indexes = []
    for p in pos:
        for n in neg:
            indexes.append((p, n))
    return indexes

class RankLoss(EvalMetric):
    """ output instance-level ranking loss, i.e. fraction of prob(y_i) < prob(y_j) where y_i is tp and y_j is fp """
    def __init__(self, threshold=0.5):
        super(AccuracyExam, self).__init__('recall_exam')
        self.threshold = threshold

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = prob2label(preds[i].asnumpy(), thresh=self.threshold)
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label, shape=1)

            for pred, l in zip(pred_label, label):
                num_pred_and_l = len(numpy.where(numpy.logical_and(pred == 1, l == 1))[0])
                num_l = len(numpy.where(l == 1)[0])
                self.sum_metric += num_pred_and_l / float(num_l) if num_l > 0 else 0

            self.num_inst += label.shape[0]


