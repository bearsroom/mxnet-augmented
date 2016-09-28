
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros((self.num_classes, ), dtype=np.float32)
        self.fp = np.zeros((self.num_classes, ), dtype=np.float32)
        self.p = np.zeros((self.num_classes, ), dtype=np.float32)
        self.tp_topk = np.zeros((self.num_classes, ), dtype=np.float32)
        self.p_topk = np.zeros((self.num_classes, ), dtype=np.float32)
        self.fp_images = [[] for _ in range(self.num_classes)]

    def update_top1(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            self.p[y_gt] += 1
            if y_pred == y_gt:
                self.tp[y_pred] += 1
            else:
                self.fp[y_pred] += 1

    def update_topk(self, pred_int_list, gt_int_list, top_k=5):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            assert len(y_pred) == top_k
            self.p_topk[y_gt] += 1
            if y_gt in y_pred:
                self.tp_topk[y_gt] += 1

    def get(self, metric='f1_score'):
        if metric == 'f1_score':
            recall = np.zeros((self.num_classes), dtype=np.float32)
            precision = np.zeros((self.num_classes), dtype=np.float32)
            f1_score = np.zeros((self.num_classes), dtype=np.float32)
            for idx in range(self.num_classes):
                if self.tp[idx] + self.fp[idx] > 0:
                    precision[idx] = self.tp[idx] / float(self.tp[idx] + self.fp[idx])
                if self.p[idx] > 0:
                    recall[idx] = self.tp[idx] / float(self.p[idx])
                if precision[idx] + recall[idx] > 0:
                   f1_score[idx] = 2 * precision[idx] * recall[idx] / float(precision[idx] + recall[idx])
            return recall, precision, f1_score
        if metric == 'topk_recall':
            recall = np.zeros((self.num_classes, ), dtype=np.float32)
            for idx in range(self.num_classes):
                if self.p_topk[idx] > 0:
                    recall[idx] = self.tp_topk[idx] / float(self.p_topk[idx])
            return recall

    def update_fp_images(self, pred_int_list, gt_int_list, probs, im_list):
        for y_pred, y_gt, prob, im_name in zip(pred_int_list, gt_int_list, probs, im_list):
            if y_gt is None:
                continue
            if y_pred != y_gt:
                prob_pred = prob[y_pred]
                prob_gt = prob[y_gt]
                self.fp_images[y_pred].append((im_name, y_gt, prob_pred, prob_gt))

    def get_fp_images(self):
        return self.fp_images


class ConfusionMatrix:
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(self.classes)
        self.reset()
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            self.matrix[y_gt, y_pred] += 1

    def get(self):
        return self.matrix

    def normalize(self):
        p = np.sum(self.matrix, axis=1)
        p[np.where(p == 0)[0]] = 1
        self.matrix = self.matrix / p

    def draw(self, output):
        plt.figure()
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix - {} classes'.format(self.num_classes))
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=90)
        plt.yticks(tick_marks, self.classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predict label')
        plt.savefig(output, format='jpg', quality=80)
        plt.clf()
