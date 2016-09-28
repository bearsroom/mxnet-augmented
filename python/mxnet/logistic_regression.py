
import mxnet as mx
import numpy as np
import logging


class KLogisticRegression(mx.operator.NumpyOp):
    """ with multiple labels possible """
    def __init__(self):
        super(KLogisticRegression, self).__init__()

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = 1.0 / (1.0 + np.exp(x))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1].astype(np.int)
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = 0.0
        for idx, labels in enumerate(l):
            for label in labels:
                if label >= 0:
                    dx[idx, label] = y[idx, label] - 1
                else:
                    dx[idx, -label] = y[idx, -label]

    def list_argument(self):
        return ['data', 'label']

    def list_output(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]
