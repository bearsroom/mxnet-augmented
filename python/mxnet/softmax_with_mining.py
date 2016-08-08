

import mxnet as mx
import numpy as np
import logging


class SoftmaxWithMining(mx.operator.NumpyOp):
    def __init__(self, mining=True, sampling=True, hard_thresh=0.5):
        super(SoftmaxWithMining, self).__init__(False)
        self.mining = mining
        self.sampling = sampling
        self.hard_thresh = hard_thresh
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
        dx[:] = y
        dx[np.arange(l.shape[0]), l] -= 1.0
        if self.mining:
            dx = self.hard_negative_mining(dx, l)

    def hard_negative_mining(self, y, l):
        pred = np.argmax(y, axis=1)
        prob = np.max(y, axis=1)
        # hard samples
        hard_tp = np.where(np.logical_and(l == pred, prob < self.hard_thresh, np.arange(l.shape[0])))[0]
        hard_fp = np.where(np.logical_and(l != pred, prob > self.hard_thresh, np.arange(l.shape[0])))[0]
        hard = np.hstack((hard_tp, hard_fp))
        if self.sampling:
            hard = self.balance_sampling(y, l, hard)
        out = np.zeros(y.shape, dtype=y.dtype)
        out[hard] = y[hard]
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
        hard_l_count, _ = np.histogram(hard_l, bins = max_l + 1)
        mean_count = np.ceil(np.mean(hard_l_count)).astype(np.int)
        for label in np.arange(max_l+1):
            if hard_l_count[label] >= mean_count:
                continue
            candidate = np.setdiff1d(np.where(l == label)[0], hard_l)
            hard = np.hstack((hard, candidate[:mean_count - hard_l_count[label]]))
        return hard

    def loss_with_negative_label(self, y, l):
        in_grad = y
        positive = np.where(l >= 0)[0]
        negative = np.where(l < 0)[0]
        # positive part is the same as original softmax loss: loss(x) = -log(exp(y)) with y the true positive label
        in_grad[positive, l[positive]] -= 1.0
        # negative part: loss(x) = -log(1-exp(y)) with y the true negative label (value y < 0)
        for neg in negative:
            in_grad_neg = in_grad[neg]
            neg_prob = in_grad_neg[l[neg]]
            in_grad_neg *= (-neg_prob / (1 - neg_prob))
            in_grad[neg] = in_grad_neg
            in_grad[neg, l[neg]] = neg_prob
        return in_grad

    def list_arguments(self):
        return ['data', 'label']

    def list_output(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

