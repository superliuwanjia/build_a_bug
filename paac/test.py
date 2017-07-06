import os
import sys
from train import get_network_and_environment_creator, bool_arg
import logger_utils
import argparse
import numpy as np
import time
import tensorflow as tf
import random
from paac import PAACLearner


sys.path.insert(0, '/home/rgoel/drmm/bug')

from dataLoader import *

def get_save_frame(name):
    import imageio

    writer = imageio.get_writer(name + '.gif', fps=30)

    def get_frame(frame):
        # print(frame.shape)

        writer.append_data(frame)

    return get_frame


def visualize(states, network_output_retina, counter = 1, name = "viz"):
    dim = (32, 32)
    print("dims",dim)
    output =  '/mnt/nvme0n1/robin/build_a_bug/paac'
    # print("gfjgidofgiodf")
    # open("random.txt","a").write(str(network_output_retina))
    print(states.shape)
    states = np.transpose(np.reshape(states,(84,84,4)), [2,0,1])
    network_output_retina = np.transpose(np.reshape(network_output_retina,(84,84,8)), [2,0,1])
    on_with_off = np.concatenate([resize_video(states,dim),\
                                  resize_video(network_output_retina[:4,:,:],dim),\
                                  resize_video(network_output_retina[4:,:,:], dim)], axis=2)
    save_video(on_with_off, \
                 os.path.join(output, name +str(counter) + ".avi"))
    print(on_with_off.shape)
    return on_with_off

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-gn', '--gif_name', default="original_game_play", type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-gne', '--gif_name_e', default=None, type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name_e")
    parser.add_argument('-gf', '--gif_folder', default='', type=str, help="The folder where to save gifs.", dest="gif_folder")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('-v', '--is_viz', default=False, type=bool, help="whether to visualise the edr or not", dest="edr_viz")
    args = parser.parse_args()
    arg_file = os.path.join(args.folder, 'args.json')
    device = args.device
    for k, v in logger_utils.load_args(arg_file).items():
        setattr(args, k, v)
    args.max_global_steps = 0
    df = args.folder
    args.debugging_folder = '/tmp/logs'
    args.device = device
    edr_name = args.gif_name_e
    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1

    args.actor_id = 0
    rng = np.random.RandomState(int(time.time()))
    args.random_seed = rng.randint(1000)

    network_creator, env_creator = get_network_and_environment_creator(args)
    network = network_creator()
    saver = tf.train.Saver()

    rewards = []
    environments = [env_creator.create_environment(i) for i in range(args.test_count)]
    if args.gif_name:
        for i, environment in enumerate(environments):
            environment.on_new_frame = get_save_frame(os.path.join(args.gif_folder, args.gif_name + str(i)))

    config = tf.ConfigProto()
    if 'gpu' in args.device:
        config.gpu_options.allow_growth = True
    print(args)
    if args.edr_viz:
        edr_outs = get_save_frame(edr_name)
    with tf.Session(config=config) as sess:
        checkpoints_ = os.path.join(df, 'checkpoints')
        network.init(checkpoints_, saver, sess)
        states = np.asarray([environment.get_initial_state() for environment in environments])
        if args.noops != 0:
            for i, environment in enumerate(environments):
                for _ in range(random.randint(0, args.noops)):
                    state, _, _ = environment.next(environment.get_noop())
                    states[i] = state

        episodes_over = np.zeros(args.test_count, dtype=np.bool)
        rewards = np.zeros(args.test_count, dtype=np.float32)
        while not all(episodes_over):
            actions, edr_output, _, _, r_s,r_m,r_l = PAACLearner.choose_next_actions_with_viz(network, env_creator.num_actions, states, sess, True)
            #red = visualize(states, edr_output)
            print("states shape", states.shape)
            print("r_s", r_s.shape)
            print("edr output shape",edr_output.shape)
                #for i in range(4):
                    # print(red[i].shape)
                    #edr_outs(red[i])
            # edr_outs(visualize(states, edr_output))
            edr_outs(r_s.reshape([84,84]))
            for j, environment in enumerate(environments):
                state, r, episode_over = environment.next(actions[j])
                states[j] = state
                rewards[j] += r
                episodes_over[j] = episode_over

        print('Performed {} tests for {}.'.format(args.test_count, args.game))
        print('Mean: {0:.2f}'.format(np.mean(rewards)))
        print('Min: {0:.2f}'.format(np.min(rewards)))
        print('Max: {0:.2f}'.format(np.max(rewards)))
        print('Std: {0:.2f}'.format(np.std(rewards)))


