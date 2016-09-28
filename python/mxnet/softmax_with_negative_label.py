

import mxnet as mx
import numpy as np
import logging


class SoftmaxWithNegative(mx.operator.NumpyOp):
    def __init__(self, mining=True, sampling=True, tp_hard_thresh=0.7, fp_hard_thresh=0.3, neg_label_grad_scale=0.5):
        super(SoftmaxWithNegative, self).__init__(False)
        self.mining = mining
        self.sampling = sampling
        self.tp_hard_thresh = tp_hard_thresh
        self.fp_hard_thresh = fp_hard_thresh
        self.neg_label_grad_scale = neg_label_grad_scale
        if self.mining:
            print('Enable softmax with hard negative mining')
            self.batch_idx = 0
            self.hard_sample_count = 0
        if self.sampling:
            print('Enable data sampling to balance samples of each category')

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        l = l.reshape((l.size, )).astype(np.int)
        y = out_data[0]
        dx = in_grad[0]
        #dx[:] = y
        #dx[np.arange(l.shape[0]), l] -= 1.0
        dx[:] = self.backward_with_negative_label(y, l)
        if self.mining:
            dx[:] = self.hard_negative_mining(dx, y, l)

    def backward_with_negative_label(self, y, l):
        in_grad = np.zeros(y.shape, dtype=y.dtype)
        in_grad[:] = y
        positive = np.where(l >= 0)[0]
        negative = np.where(l < 0)[0]
        # positive part is the same as original softmax loss: loss(x) = -log(exp(x_y)) with y the true positive label
        in_grad[positive, l[positive]] -= 1.0
        # negative part: loss(x) = -log(1-exp(x_y)) with y the true negative label (value y < 0)
        for neg in negative:
            neg_prob = in_grad[neg, -l[neg]]
            in_grad[neg] *= (-neg_prob / (1 - neg_prob)) * self.neg_label_grad_scale
            in_grad[neg, -l[neg]] = neg_prob * self.neg_label_grad_scale
            #print('in_grad: {}'.format(in_grad[neg]))
        return in_grad


    def get_hard_samples(self, preds, labels, probs, mode='pos'):
        if mode == 'pos':
            # hard samples with positive labels
            hard_tp = np.where(np.logical_and(l == pred, prob < self.tp_hard_thresh, np.arange(l.shape[0])))[0]
            hard_fp = np.where(np.logical_and(l != pred, prob > self.fp_hard_thresh, np.arange(l.shape[0])))[0]
            hard = np.hstack((hard_tp, hard_fp))
        elif mode == 'neg':
            # hard negative samples with negative labels (l < 0)
            hard = np.where(np.logical_and(np.abs(l) == pred, prob > self.fp_hard_thresh, np.arange(l.shape[0])))[0]
        else:
            raise ValueError('Mode invalid: {}, expect mode=pos/neg'.format(mode))
        return hard


    def hard_negative_mining(self, in_grad, y, l):
        pred = np.argmax(y, axis=1)
        prob = np.max(y, axis=1)

        positive = np.where(l >= 0)[0]
        negative = np.where(l < 0)[0]
        hard_positive = positive[self.get_hard_samples(pred[positive], l[positive], prob[positive])]
        hard_negative = negative[self.get_hard_samples(pred[negative], l[negative], prob[negative], mode='neg')]
        hard = np.hstack((hard_positive, hard_negative))

        if self.sampling:
            hard = self.balance_sampling(y, l, hard)
        out = np.zeros(in_grad.shape, dtype=in_grad.dtype)
        out[hard] = in_grad[hard]
        self.hard_sample_count += hard.size
        self.batch_idx += 1
        if self.batch_idx % 200 == 0 and self.batch_idx != 0:
            mean_hard_count = self.hard_sample_count / float(200)
            logging.info('Trained {} batches, use {}/{} hard samples in average'.format(self.batch_idx, mean_hard_count, l.size))
            self.hard_sample_count = 0
        return out

    def balance_sampling(self, y, l, hard):
        # count label frequency
        max_l = np.max(l)
        hard_l = l[hard]
        hard_l_count, _ = np.histogram(np.abs(hard_l), bins = max_l + 1)
        mean_count = np.ceil(np.mean(hard_l_count)).astype(np.int)
        for label in np.arange(max_l+1):
            if hard_l_count[label] >= mean_count:
                continue
            candidate = np.setdiff1d(np.where(np.abs(l) == label)[0], hard_l)
            hard = np.hstack((hard, candidate[:mean_count - hard_l_count[label]]))
        return hard

    def list_arguments(self):
        return ['data', 'label']

    def list_output(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

