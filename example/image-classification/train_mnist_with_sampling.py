import find_mxnet
import mxnet as mx
from mxnet.data_sampling_iter import DataSamplingIter
import argparse
import os, sys
import struct
import train_model
import numpy as np

def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip")
        os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def load_mnist_to_array(dataset, data_dir, network):
    if dataset == 'training':
        fname_img = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    elif dataset == 'testing':
        fname_img = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError('dataset must be testing or training')

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
        #lbl = lbl.reshape(lbl.shape[0], 1)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        if network == 'mlp':
            img = np.fromfile(fimg, dtype=np.uint8).reshape(lbl.shape[0], 1, rows * cols)
        elif network == 'lenet':
            img = np.fromfile(fimg, dtype=np.uint8).reshape(lbl.shape[0], 1, rows, cols)

    return img, lbl


def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir
        if '://' not in args.data_dir:
            _download(args.data_dir)
        flat = False if len(data_shape) == 3 else True

        train_img, train_lbl = load_mnist_to_array('training', data_dir, args.network)
        test_img, test_lbl = load_mnist_to_array('testing', data_dir, args.network)

        train           = DataSamplingIter(
            data        = train_img,
            label       = train_lbl,
            batch_size  = args.batch_size,
            shuffle     = True,
            sampling    = True,
            num_sample_per_label = 8000)

        val             = DataSamplingIter(
            data        = test_img,
            label       = test_lbl,
            batch_size  = args.batch_size,
            sampling    = False)

        return (train, val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp', 'lenet'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp()
    else:
        data_shape = (1, 28, 28)
        net = get_lenet()

    # train
    train_model.fit(args, net, get_iterator(data_shape))
