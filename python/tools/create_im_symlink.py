
import os
import re
import sys
import argparse

def create_symbol_link(src_dir, images, dst_root, classes):
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)
    for cls, im_cls in zip(classes, images):
        if im_cls == []:
            continue
        dst_dir = os.path.join(dst_root, cls)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        for im in im_cls:
            src = os.path.join(src_dir, im)
            dst = os.path.join(dst_dir, im)
            os.symlink(src, dst)


def parse_images_file(image_file, classes):
    images_list = open(image_file).read().splitlines()
    images_list = [im.split() for im in images_list]
    print('Will create {} symbol links'.format(len(images_list)))

    images = [[] for _ in range(len(classes))]
    for im in images_list:
        idx = classes.index(im[1])
        images[idx].append(im[0])

    return images


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Give a list of images with label, create symbol links for these images')
    parser.add_argument('--img-list', dest='im_list', required=True,
                        help='List of images with label to generate symbol links',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes', required=True,
                        help='List of classes',
                        default=None, type=str)
    parser.add_argument('--src-dir', dest='src_dir', required=True,
                        help='Image data source dir to join the paths',
                        default=None, type=str)
    parser.add_argument('--dst-dir', dest='dst_dir', required=True,
                        help='Destination root dir to store symbolink',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    classes = open(args.classes).read().splitlines()
    classes = [c[-1] for c in classes]
    images = parse_images_file(args.img_list, classes)
    create_symbol_link(args.src_dir, images, args.dst_dir, classes)
