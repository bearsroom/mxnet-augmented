
import find_mxnet
import mxnet as mx
import argparse
import sys

"""
script to copy params of src symbol to dst symbol
src symbol and dst symbol may have different argument names
warning: make sure that the structures of 2 symbols are nearly the same!
this copy is NOT based on argument names
"""


def copy(src_sym_prefix, dst_sym_prefix, src_epoch, dst_epoch, overwrite=False):
    # load 2 symbols
    sym1, arg_params1, aux_params1 = mx.model.load_checkpoint(src_sym_prefix, src_epoch)
    sym2, arg_params2, aux_params2 = mx.model.load_checkpoint(dst_sym_prefix, dst_epoch)

    arg_names1 = sym1.list_arguments()
    arg_names2 = sym2.list_arguments()

    copied = []
    for name1, name2 in zip(arg_names1, arg_names2):
        # copy arg_params1[name1] to arg_params2[name2]
        if name1 not in arg_params1.keys() or name2 not in arg_params2.keys():
            continue
        if arg_params1[name1].shape == arg_params2[name2].shape:
            arg_params2[name2] = arg_params1[name1]
            copied.append((name1, name2))

    f = open(dst_sym_prefix + '_copy_log', 'w')
    for copy_pair in copied:
        print('{} -> {}'.format(copy_pair[0], copy_pair[1]))
        f.write('{} -> {}\n'.format(copy_pair[0], copy_pair[1]))

    # save symbol2
    if not overwrite:
        dst_sym_prefix = dst_sym_prefix + '_copied'
    mx.model.save_checkpoint(dst_sym_prefix, 1, sym2, arg_params2, aux_params2)
    print('Saved new params file to %s-%04d.params' % (dst_sym_prefix, 1))
    f.write('Saved new params file to %s-%04d.params' % (dst_sym_prefix, 1))


def parse_args():
    parser = argparse.ArgumentParser(description='copy src symbol params to dst symbol params')
    parser.add_argument('--src-sym', type=str, required=True,
                        help='source symbol file prefix')
    parser.add_argument('--src-epoch', type=int, required=True,
                        help='source epoch to load')
    parser.add_argument('--dst-sym', type=str, required=True,
                        help='dest symbol file prefix')
    parser.add_argument('--dst-epoch', type=int, required=True,
                        help='dest epoch to load')
    parser.add_argument('--overwrite', action='store_true', default=bool,
                        help='overwrite exist params if exists')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    copy(args.src_sym, args.dst_sym, args.src_epoch, args.dst_epoch, args.overwrite)
