import tensorflow as tf
import custom.utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Who knows what this file will do in the end.")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    with tf.Session() as sess:

        custom.utils.load_model(args.model_dir)
        custom.utils.print_trainable_variables()
        convWeights0 = custom.utils.extract_params(1)
        #print(convWeights0)