import argparse
import gym
import os
import numpy as np
import sys
# directory: ~/repo1/atari-state-representation-learning/custom

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn

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
    parser.add_argument("--randomPolicy", type=int, default=None, help="Collecting experience using 0:random policy, 1:expert policy and 2:mixing .")
    boolean_flag(parser, "stochastic", default=True,
                 help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def int_buffers():
    return {'frames': [],
           'finConvAct': [],
           'fstFullAct': [],
           'action': [],
           'reward': []}


def save_step(buffers, inp):
    """
    Saves states and activities after each frame
    """
    for key, value in buffers.items():
        value.append(inp[key])


def save_buffers_to_file(buffers, ssdir, num_episode):
    """
    Saves the buffers to corresponding files after an episodes ends
    """

    out_dir = os.path.join(ssdir, 'exp_in_total_%03i_episodes' % num_episode)

    try:
        os.makedirs(out_dir)
    except OSError:
        if not os.path.exists(ssdir):
            sys.exit("Error could't make output dir")

    for key, value in buffers.items():
        np.save(os.path.join(out_dir, "%s.npy" % key), np.array(value))


def play(env, act, stochastic, video_path, ssdir, maxNumEpisodes, random_policy = 1):
    buffers = int_buffers() # Initialization of  buffer
    
    # variable 'random_policy' is to decide if to collect random experiencwe with CNN-induced features
    
    # Jun5. Kuan: I like to make the code flexible s.t. 
    #   1. I can collect experience (with features) using random policy 
    #       1.1 Later, I also like it to collect experience during learning through on-policy approach (for ex, A-C)
    #   2. I can collect experience altogether regardless of the episodes
    
    
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()

    while True:
        env.unwrapped.render()
        video_recorder.capture_frame()
        
        # Jun5. Modification1: collecting random experience
        # is there more efficient way to code this functionality of if's ?
        if random_policy == 0:            
            _, finConvAct, fstFullAct = act(np.array(obs)[None], stochastic=stochastic)
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            
        elif random_policy == 1:
            action, finConvAct, fstFullAct = act(np.array(obs)[None], stochastic=stochastic)
            obs, rew, done, info = env.step(action)                    
            
        elif random_policy == 2:
            if np.random.uniform(0,1) > 0.3:
                action, finConvAct, fstFullAct = act(np.array(obs)[None], stochastic=stochastic)
                obs, rew, done, info = env.step(action)                               
            else:
                _, finConvAct, fstFullAct = act(np.array(obs)[None], stochastic=stochastic)
                action = env.action_space.sample()
                obs, rew, done, info = env.step(action)                

        # Save states and activities in buffers for current episode            
        save_step(buffers,
                  {'frames': env.render('rgb_array'),
                   'finConvAct': np.array(finConvAct),
                   'fstFullAct': np.array(fstFullAct),
                   'action': action,
                   'reward': rew})
        if done:
            obs = env.reset()
            
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"]) # I guess here is kind of adding num_episodes when each episode end!
            if num_episodes >= int(maxNumEpisodes):
                # Save buffered states and activities altogether to file afer all the episodes ends
                save_buffers_to_file(buffers, ssdir, int(maxNumEpisodes))
                return    
        

if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()

        maxNumEpisodes = args.maxNumEpisodes
        if not maxNumEpisodes:
            maxNumEpisodes = 0

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
        play(env, act, args.stochastic, args.video, ssdir, maxNumEpisodes, args.randomPolicy)
