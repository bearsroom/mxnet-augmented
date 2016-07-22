# pylint: skip-file

import sys
import os
from io import NDArrayIter
import numpy as np
import logging


class DataSamplingIter(NDArrayIter):
    """
    this data iter class provides some data sampling methods:
    - Downsampling
    - Upsampling
    - Shuffling
    """
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad', sampling=True, num_sample_per_label=5000):
        super(DataSamplingIter, self).__init__(data, label=label, batch_size=batch_size, shuffle=shuffle, last_batch_handle=last_batch_handle)
        self.shuffle = shuffle
        self.label_sampling_rates = 0.0

        # sampling configuration
        if sampling:
            self.sampling = self._get_sampling_rates(num_sample_per_label=num_sample_per_label)
        else:
            self.sampling = False
        if self.sampling:
            logging.info('Data set has {} images of {} classes'.format(self.data[0][1].shape[0], np.max(self.label[self.pos_label][1]+1)))
            logging.info('Sampling enabled, random sample {} samples for each class at the beginning of each epoch'.format(num_sample_per_label))
            self.do_sampling()


    def _get_sampling_rates(self, label_name='softmax', num_sample_per_label=5000):
        label = []
        for idx, pair in enumerate(self.label):
            if pair[0] == label_name:
                label = pair[1]
                self.pos_label = idx
        if label:
            max_label = np.max(label)
            self.label_count, _ = np.histogram(label, bins = max_label + 1)
            assert self.label_count.shape[0] == max_label + 1
            self.label_sampling_rates = 5000 / self.label_count.astype(np.float)
            return True
        else:
            logging.warning('No such label {} in self.label, self.label names: {}'.format(label_name, ' '.join([l[0] for l in self.label])))
            logging.warning('Will not perform sampling due to label name error')
            return False


    def _down_sampling(self, label_idx, label_name='softmax'):
        rate = self.label_sampling_rates[label_idx]
        label = self.label[self.pos_label][1]
        data_idx = np.where(label == label_idx)[0]
        rnd = np.random.rand(data_idx.shape)
        rnd_idx = data_idx[np.where(np.random.rand(data_idx.shape[0]) < rate)[0]]
        return rnd_idx


    def _up_sampling(self, label_idx, label_name='softmax'):
        rate = self.label_sampling_rates[label_idx]
        repeat = np.floor(rate)
        rate = rate - repeat
        label = self.label[self.pos_label][1]
        data_idx = np.where(label == label_idx)[0]
        repeat_idx = np.repeat(data_idx, repeat)
        rnd_idx = data_idx[np.where(np.random.rand(data_idx.shape[0]) < rate)[0]]
        return np.hstack((repeat_idx, rnd_idx))


    def do_sampling(self, label_name='softmax'):
        allow_data_idx = np.empty((0,))
        for label_idx in np.range(len(self.label_sampling_rates)):
            if self.label_count[label_idx] == 0: # we have no data with label label_idx
                continue
            if self.label_sampling_rates[label_idx] >= 1.0:
                allow_data_idx = np.hstack((allow_data_idx, _up_sampling(label_idx, label_name)))
            else:
                allow_data_idx = np.hstack((allow_data_idx, _down_sampling(label_idx, label_name)))
        np.random.shuffle(allow_data_idx)
        self.sample_idx = allow_data_idx
        self.num_data = self.sample_idx.shape[0]


    def do_shuffling(self):
        if self.shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]


    def getdata(self):
        if self.sampling:
            data = [(k, v[self.sample_idx]) for k, v in self.data]
            return self._getdata(data)
        else:
            return self._getdata(self.data)


    def getlabel(self):
        if self.sampling:
            label = [(k, v[self.sample_idx]) for k, v in self.label]
            return self._getdata(label)
        else:
            return self._getdata(self.label)


    def reset(self):
        if self.sampling:
            self.do_sampling()
        elif self.shuffle:
            self.do_shuffling()
        super(DataSamplingIter, self).reset()
