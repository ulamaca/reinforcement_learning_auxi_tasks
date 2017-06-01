import argparse
import gym
import os
import numpy as np
import sys

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn

import matplotlib as mpl

mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

from custom.model import model, dueling_model
from custom.build_graph import build_act


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to mp4 file where the video of first episode will be recorded.")
    parser.add_argument("--saveStateDir", type=str, default=None, help="Save states in dir.")
    parser.add_argument("--maxNumEpisodes", type=str, default=None, help="Maximal number of episodes.")
    boolean_flag(parser, "stochastic", default=True,
                 help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def plot_final_conv_act(f_bf, a_bf):
    fig = plt.figure()
    for i in range(0, int(f_bf.shape[0] * 2), 2):
        fig.add_subplot(int(f_bf.shape[0]), 2, np.floor(i / 2) * 2 + 1)
        plt.imshow(f_bf[int(i / 2)])
        fig.add_subplot(int(f_bf.shape[0]), 2, np.floor(i / 2) * 2 + 2)
        act_size = a_bf[int(i / 2)].shape[0]
        act = np.reshape(a_bf[int(i / 2)], (int(np.sqrt(act_size)), int(np.sqrt(act_size))))
        plt.imshow(act)
    plt.show()

def save_step(frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer,
              env, finConvAct, fstFullAct, action, rew):
    frame_buffer.append(env.render('rgb_array'))
    finConvAct_buffer.append(finConvAct)
    fstFullAct_buffer.append(fstFullAct)
    action_buffer.append(action)
    reward_buffer.append(rew)

def save_buffers_to_file(frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer,
                         ssdir, num_episode):
    out_dir = os.path.join(ssdir, 'episode_%03i' % num_episode)

    try:
        os.makedirs(out_dir)
    except OSError:
        if not os.path.exists(ssdir):
            sys.exit("Error could't make output dir")

    np.save(os.path.join(out_dir, "frames.npy"), np.array(frame_buffer))
    frame_buffer = []

    np.save(os.path.join(out_dir, "finConvAct.npy"), np.array(finConvAct_buffer))
    finConvAct_buffer = []

    np.save(os.path.join(out_dir, "fstFullAct.npy"), np.array(fstFullAct_buffer))
    fstFullAct_buffer = []

    np.save(os.path.join(out_dir, "actions.npy"), np.array(action_buffer))
    action_buffer = []

    np.save(os.path.join(out_dir, "rewards.npy"), np.array(reward_buffer))
    reward_buffer = []

    return frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer

def play(env, act, stochastic, video_path, ssdir, maxNumEpisodes):
    frame_buffer = []
    finConvAct_buffer = []
    fstFullAct_buffer = []
    action_buffer = []
    reward_buffer = []

    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    while True:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action, finConvAct, fstFullAct = act(np.array(obs)[None], stochastic=stochastic)
        obs, rew, done, info = env.step(action)
        save_step(frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer,
                  env, finConvAct, fstFullAct, action, rew)
        if done:
            obs = env.reset()
            frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer = \
                save_buffers_to_file(frame_buffer, finConvAct_buffer, fstFullAct_buffer, action_buffer, reward_buffer,
                                 ssdir, num_episodes)
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])
            if num_episodes >= int(maxNumEpisodes):
                return

if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()

        model_name = os.path.splitext(os.path.basename(args.model_dir))[0]
        ssdir = args.saveStateDir
        if not ssdir:
            sys.exit('Please provide a name for the output dir')
        try:
            ssdir = os.path.join(ssdir, model_name)
            os.makedirs(ssdir)
        except OSError:
            if not os.path.exists(ssdir):
                sys.exit("Error could't make output dir")

        env = make_env(args.env)
        act = build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, act, args.stochastic, args.video, ssdir, args.maxNumEpisodes)
