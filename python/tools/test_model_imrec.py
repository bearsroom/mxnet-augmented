
import find_mxnet
import mxnet as mx
import logging
import argparse
import os

classes = ['none', 'plant', 'animal', 'landscape', 'nightscape', 'construction', 'way', 'furnishing', 'clothing',
           'text', 'painting', 'transport', 'food', 'container', 'musical_instrument', 'selfie', 'statue', 'e_device', 'toy']


parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
parser.add_argument('--data-dir', type=str, required=True,
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--classes', type=str, required=True,
                    help='Class list')
parser.add_argument('--num-class', type=int, default=19,
                    help='number of classes / output labels')
parser.add_argument('--num-batch', type=int, default=None,
                    help='number of batch to test, if not set, will test the whole dataset')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-file', type=str,
		    help='the name of log file')
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')
parser.add_argument('--test-dataset', type=str, default="val.rec",
                    help="test dataset name")
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
parser.add_argument('--eval-metrics', type=str, default='recall,precision,acc',
                    help='metrics for evaluation, e.g "recall,precision,acc"')
args = parser.parse_args()


def set_logger(args, kv):
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)


def get_iterator(args, kv):
    data_shape = (3, args.data_shape, args.data_shape)
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.test_dataset),
        mean_img = os.path.join(args.data_dir, 'test_mean.nd'),
        #mean_r      = 123.68,
        #mean_g      = 116.779,
        #mean_b      = 103.939,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return test


def load_model(args, kv):
    # load model
    model_prefix = args.model_prefix
    if model_prefix is None:
        logging.fatal('No model prefix provided!')
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        model = mx.model.FeedForward.load(model_prefix, args.load_epoch)
    return model


def test(args):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # data iterator
    test_iter = get_iterator(args, kv)

    set_logger(args, kv)

    # network
    model = load_model(args, kv)

    eval_metrics = args.eval_metrics.split(',')
    kwargs = {}
    if set(eval_metrics) & set(['recall', 'precision']):
        kwargs = {'num_classes': args.num_class}
    print('Evalutaion metrics: {}'.format(eval_metrics))
    results = model.score(test_iter, eval_metric=eval_metrics, num_batch=args.num_batch, batch_end_callback=None, reset=True, **kwargs)
    logging.info('======================= Test Results ========================')
    for idx, cls in enumerate(classes):
        if 'acc' in eval_metrics:
            acc_idx = eval_metrics.index('acc')
            acc = results[acc_idx]
            eval_metrics.pop(acc_idx)
            results.pop(acc_idx)
            logging.info('Total accuracy: {:>12}'.format(acc))
        cls_result = [metric+': '+str(res[idx]) for metric, res in zip(eval_metrics, results)]
        logging.info('class {:>12}, {:>12}'.format(cls, ', '.join(cls_result)))

classes = open(args.classes).read().splitlines()
classes = [cls.split()[0] for cls in classes]
args.num_classes = len(classes)

test(args)
