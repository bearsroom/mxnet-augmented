
import find_mxnet
import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import argparse
import sys
import os
import time
import logging
from data_loader import preprocess
from metrics import Metrics
import cPickle as pkl


def load_model(model_prefix, num_epoch, batch_size, gpu_id=0):
    model = mx.model.FeedForward.load(model_prefix, num_epoch, ctx=mx.gpu(gpu_id), numpy_batch_size=batch_size)
    # plot the network
    #mx.viz.pot_network(model.symbol, shape={"data": (1, 3, 224, 224)})
    return model


def get_partial_symbol(top_feat_name, model, batch_size, gpu_id=0):
    internals = model.symbol.get_internals()
    try:
        feat_symbol = internals[top_feat_name]
        feat_extractor = mx.model.FeedForward(ctx=mx.gpu(gpu_id), symbol=feat_symbol, numpy_batch_size=batch_size, arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
        return feat_extractor
    except Exception, e:
        logging.error('Cannot get symbol with top feature {}: {}'.format(top_feat_name, e))
        return None


def load_batch(data_dir, im_list, mean_img_file, batch_size=200):
    batch_size = min(batch_size, len(im_list))
    # load mean image
    mean_img = mx.nd.load(mean_img_file)['mean_img'].asnumpy()
    data_shape = (batch_size, 3, 224, 224)
    batch = np.zeros(data_shape, dtype=np.float)

    idx = 0
    for line in im_list:
        im_name = line.split()[0]
        im_path = os.path.join(data_dir, im_name)
        img = io.imread(im_path)
        try:
            batch[idx] = preprocess(img, mean_img)
            idx += 1
        except Exception, e:
            logging.error('{}: failed to load: {}'.format(im_name, e))
        if idx >= batch_size:
            break
    return batch[:idx]


def predict(model, batch):
    feat = model.predict(batch)
    return feat


def get_feats(feat_list, model_prefix, num_epoch, batch, gpu_id=0):
    batch_size = batch.shape[0]
    logging.info('Loading model {}, epoch {}'.format(model_prefix, num_epoch))
    full_model = load_model(model_prefix, num_epoch, batch_size, gpu_id)

    output = {}
    mean = {}
    std = {}
    for feat in feat_list:
        if not feat.endswith('_output'):
            feat += '_output'
        feat_extractor = get_partial_symbol(feat, full_model, batch_size, gpu_id)
        if feat_extractor is None:
            continue
        logging.info('Getting feature {}, batch size {}'.format(feat, batch_size))
        output[feat] = predict(feat_extractor, batch)
        mean[feat] = np.mean(output[feat], axis=(0,1,2,3)[1:output[feat].ndim])
        std[feat] = np.std(output[feat], axis=(0,1,2,3)[1:output[feat].ndim])
        logging.info('Feature {} has shape {}, mean value {}, mean std {}, max std {}'.format(feat, output[feat].shape, np.mean(mean[feat]), np.mean(std[feat]), np.max(std[feat])))
    return output, mean, std


def test(data_dir, im_list, feat_list, mean_img_file, model_prefix, num_epoch, output_pkl, batch_size=200, gpu_id=0):
    start = time.time()
    batch = load_batch(data_dir, im_list, mean_img_file, batch_size)
    elapsed = time.time() - start
    logging.info('Batch loaded, batch size {}, elapsed {}s'.format(batch_size, elapsed))

    output, mean, std = get_feats(feat_list, model_prefix, num_epoch, batch, gpu_id)
    if output != {} and output_pkl is not None:
        with open(output_pkl, 'wb') as f:
            pkl.dump({'feat': output, 'mean': mean, 'std': std}, f)
        logging.info('Saved output to {}'.format(output_pkl))
    else:
        logging.warning('No output file created')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='GoogLeNet classifier: given a single image, return its label')
    parser.add_argument('--img-list', dest='im_list', required=True,
                        help='List of images to predict',
                        default=None, type=str)
    parser.add_argument('--feat', dest='feat', required=True,
                        help='Names or list of features names to extract, if provided in stream, use "," as seperator',
                        default=None, type=str)
    parser.add_argument('--data-dir', dest='data_dir', required=True,
                        help='Image data dir to join the paths',
                        default=None, type=str)
    parser.add_argument('--model-prefix', dest='model_prefix', required=True,
                        help='GoogLeNet model prefix',
                        default=None, type=str)
    parser.add_argument('--num-epoch', dest='num_epoch', required=True,
                        help='num of epoch to load',
                        default=None, type=int)
    parser.add_argument('--mean-img', dest='mean_img', required=True,
                        help='Mean image to substract',
                        default=None, type=str)
    parser.add_argument('--batch-size', type=int,
                        help='Number of images to test',
                        default=200)
    parser.add_argument('--gpus', dest='gpus',
                        help="GPU devices to use, split by ','",
                        default='0', type=str)
    parser.add_argument('--output', dest='output',
                        help='Output file , results will be stored in pickle format',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global pid
    pid = os.getpid()
    args = parse_args()

    FORMAT = "%(asctime)s %(levelname)s %(process)d %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO, filename=None)

    im_list = open(args.im_list).read().splitlines()

    if os.path.isfile(args.feat):
        feat_list = open(args.feat).read().splitlines()
    else:
        feat_list = args.feat.split(',')

    gpus = int(args.gpus.split(',')[0])
    test(args.data_dir, im_list, feat_list, args.mean_img, args.model_prefix, args.num_epoch, args.output, args.batch_size, gpus)
