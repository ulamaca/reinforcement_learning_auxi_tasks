from itertools import chain
import tensorflow as tf
import os.path

def parse_range(rng):
    parts = rng.split('-')
    if 1 > len(parts) > 2:
        raise ValueError("Bad range: '%s'" % (rng,))
    parts = [int(i) for i in parts]
    start = parts[0]
    end = start if len(parts) == 1 else parts[1]
    if start > end:
        end, start = start, end
    return range(start, end + 1)

def parse_range_list(rngs):
    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))

def load_model(dir):
    sess = tf.get_default_session()

    fname_meta = os.path.join(dir, 'saved.meta')
    fname_params = os.path.join(dir, 'saved')

    saver = tf.train.import_meta_graph(fname_meta)
    saver = saver.restore(sess, fname_params)

def print_trainable_variables():
    for var_index in range(0, len(tf.trainable_variables())):
        print('%20i: %s' % (var_index, tf.trainable_variables()[var_index].name))

def extract_params(index):
    sess = tf.get_default_session()
    return sess.run(tf.trainable_variables()[index])