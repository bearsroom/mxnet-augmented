
"""
DataIter for joint training of softmax and klogistic regression
this dataiter will provide 2 sets of labels:
one for softmax, i.e label[sample_idx] = label ( >= 0 for positive label, < 0 for negative label)
another for klogistic regression, i.e label[sample_idx] = [label0, label1, ..., label_n] and label_i = {-1(drop), 0(negative), 1(positive)}
need multi-label list as input label list
currently only support single positive label for softmax
"""

import sys, os
from .io import DataIter, DataBatch
import mxnet as mx


class SoftmaxKLogisticDataIter(DataIter):
    def __init__(self, data_iter, softmax_pos_only=True):
        super(SoftmaxKLogisticDataIter, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.softmax_pos_only = softmax_pos_only
        if not self.softmax_pos_only:
            raise NotImplementedError("Currently only support single positive label for softmax")

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        return [('softmax_label', provide_label[1][:-1]), \
                ('logistic_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return DataBatch(data=batch.data, label=[self.multi2single(label), label], \
                pad=batch.pad, index=batch.index)

    def multi2single(self, multi_label):
        softmax_label = mx.nd.zeros((multi_label.shape[0], ))
        if self.softmax_pos_only:
            for i in range(multi_label.shape[0]):
                for j in range(multi_label.shape[1]):
                    if multi_label[i, j] > 0:
                        softmax_label[i] = j # currently only use the first positive label
                        break
        else:
            raise NotImplementedError("Currently only support single positive label for softmax")
        return softmax_label
