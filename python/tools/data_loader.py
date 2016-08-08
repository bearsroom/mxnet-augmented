
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
from multiprocessing import Process, Queue, Lock

processes = []
LOAD_CAPACITY = 2000

def preprocess(img, mean_img, crop_mode='center'):
    # crop image
    if crop_mode == 'random':
        short_edge = min(img.shape[:2])
        yy = max(np.random.randint(img.shape[0] - short_edge + 1) - 1, 0)
        xx = max(np.random.randint(img.shape[1] - short_edge + 1) - 1, 0)
        crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    else:
        if crop_mode != 'center':
            logging.warning('Currently we provide only random crop and center crop, use center crop by default')
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy:yy+short_edge, xx:xx+short_edge]

    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))

    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_img
    normed_img.resize(1, 3, 224, 224)
    return normed_img


def preprocess_worker(proc_id, im_list, data_dir, mean_img_file, out_que, lock, gt_mode=False):
    # load mean image
    mean_img = mx.nd.load(mean_img_file)['mean_img'].asnumpy()

    idx = 0
    num_field = len(im_list[0].split())
    start = time.time()
    for line in im_list:
        if gt_mode:
            im_name, gt_label = line.split()[:2]
            gt_label = int(gt_label)
        else:
            im_name = line.split()[0]
            gt_label = None
        try:
            # load image
            im_path = os.path.join(data_dir, im_name)
            img = io.imread(im_path)
            # sanity check
            if len(img.shape) != 3:
                logging.warning('{}: not in color mode, drop'.format(im_name))
                continue
            elif img.shape[-1] != 3:
                logging.warning('{}: color image but nor in RGB mode, drop'.format(im_name))
                continue
            img = preprocess(img, mean_img)
            lock.acquire()
            out_que.put((im_name, img, gt_label))
            lock.release()
            idx += 1
            if idx % 1000 == 0 and idx != 0:
                elapsed = time.time() - start
                logging.info('Preprocessor #{} processed {} images, elapsed {}s'.format(proc_id, idx, elapsed))
        except Exception, e:
            logging.error('{}: {}'.format(im_name, e))

    lock.acquire()
    out_que.put('FINISH')
    lock.release()
    elapsed = time.time() - start
    logging.info('Preprocessor #{} finished, processed {} images, elapsed {}s'.format(proc_id, idx, elapsed))
    return


class BatchLoader():
    def __init__(self, im_list, data_dir, mean_img_file, num_preprocessor, out_que_list, batch_size=100, gt_mode=False):
        self.num_preprocessor = num_preprocessor
        self.batch_size = batch_size
        self.out_que_list = out_que_list
        # distribute im_list
        im_proc_list = [[] for _ in range(num_preprocessor)]
        for idx, line in enumerate(im_list):
            which_idx = idx % num_preprocessor
            im_proc_list[which_idx].append(line)

        self.processes = []
        self.que = Queue(LOAD_CAPACITY) # at most 5000 images in queue
        self.lock = Lock()
        for idx in range(num_preprocessor):
            p = Process(target=preprocess_worker,
                        args=(idx, im_proc_list[idx], data_dir, mean_img_file, self.que, self.lock, gt_mode))
            p.daemon = True
            self.processes.append(p)

    def start_fetcher(self):
        for p in self.processes:
            p.start()

    def provide_batch(self):
        batch_idx = 0
        while True:
            try:
                batch = self.get()
            except MemoryError, e:
                logging.info('BatchLoader: {}'.format(e))
                for que in self.out_que_list:
                    que.put('FINISH')
                logging.error('BatchLoader will exit due to unexpected MemoryError :(')
                break
            if batch:
                for que in self.out_que_list:
                    que.put(batch)
                    batch_idx +=1
                    if batch_idx % 10 == 0 and batch_idx != 0:
                        logging.info('BatchLoader send {} batches of {} images'.format(batch_idx, self.batch_size))
            else:
                for que in self.out_que_list:
                    que.put('FINISH')
                logging.info('BatchLoader exit...')
                break

    def get_processes(self):
        if len(self.processes) == 0:
            raise ValueError('No internal processes!')
        return self.processes

    def get(self):
        batch = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float)
        im_names = [None] * self.batch_size
        gt = [None] * self.batch_size
        if self.num_preprocessor > 0:
            idx = 0
            while idx < self.batch_size:
                data = self.que.get()
                if data == 'FINISH':
                    self.num_preprocessor -= 1
                    if self.num_preprocessor <= 0:
                        logging.info('All preprocessors terminated')
                        break
                    else:
                        continue
                im_names[idx] = data[0]
                batch[idx] = data[1]
                gt[idx] = data[2]
                idx += 1
            return (im_names, batch, gt)
        else:
            logging.warning('No batch left')
            return None

    def __del__(self):
        for p in self.processes:
            p.join()
        logging.info('BatchLoader terminated')


