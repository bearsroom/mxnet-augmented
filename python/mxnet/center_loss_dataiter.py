
"""
DataIter for Center Loss
"""

import sys, os
from .io import DataIter, DataBatch


class CenterLossDataIter(DataIter):
    def __init__(self, data_iter):
        super(CenterLossDataIter, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        return [('softmax_label', provide_label[1]), \
                ('center_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return DataBatch(data=batch.data, label=[label, label], \
                pad=batch.pad, index=batch.index)

