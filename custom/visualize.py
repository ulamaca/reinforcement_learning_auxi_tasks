import argparse
import os
import sys

import numpy as np

import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

from custom.utils import parse_range_list


def parse_args():
    parser = argparse.ArgumentParser("Visualize extracted data (states and activities.")

    parser.add_argument("--buffersDir", type=str, required=True, help="Name of the dir that containes the saved buffers")
    parser.add_argument("--episode", type=int, default=0, required=True, help="Load the following episode. ")
    parser.add_argument("--frames", type=str, default=None, required=True, help="Load the following frames use '-' and ','.")

    return parser.parse_args()


def load_buffers(path):
    buffers_names = [os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    buffers = dict()

    for buffer_name in buffers_names:
        buffers[buffer_name] = np.load(os.path.join(path, buffer_name + '.npy'))

    return buffers


def show_activities(buffers, disp_frames):
    fig = plt.figure()
    columns = 3

    num_frames = len(disp_frames)

    for i in range(0, num_frames):
        fig.add_subplot(num_frames, columns, i*columns+1)
        plt.imshow(np.array(buffers['frames'][disp_frames[i]]))
        fig.add_subplot(num_frames, columns, i * columns + 2)
        plt.imshow(act2d(np.array(buffers['finConvAct'][disp_frames[i]])))
        fig.add_subplot(num_frames, columns, i * columns + 3)
        plt.bar(range(0, 512), np.array(buffers['fstFullAct'][disp_frames[i]])[0])

    plt.show()


def act2d(activities):
    size = activities.shape[1]
    ssize = int(np.sqrt(size))

    return np.reshape(activities, (ssize, ssize))


if __name__ == '__main__':
    args = parse_args()

    disp_frames = parse_range_list(args.frames)
    episode = '%03i' % args.episode
    buffers_path = os.path.join(args.buffersDir, 'episode_%s' % episode)

    if not os.path.exists(buffers_path):
        sys.exit("BuffersDir doesn't exist.")

    buffers = load_buffers(buffers_path)
    show_activities(buffers, disp_frames)

    print('done')
