#!/usr/bin/env python
#
import sys
import numpy as np
import os
import random
import replay
import time
import argparse
import dqn
from atari_environment import AtariEnvironment
from state import State
import tensorflow as tf
import shutil
import pdb


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch-steps", type=int, default=250000, help="how many steps (=4 frames) to run during a training epoch (approx -- will finish current game)")
parser.add_argument("--eval-epoch-steps", type=int, default=125000, help="how many steps (=4 frames) to run during an eval epoch (approx -- will finish current game)")
parser.add_argument("--replay-capacity", type=int, default=1000000, help="how many states to store for future training")
parser.add_argument("--prioritized-replay", action='store_true', help="Prioritize interesting states when training (e.g. terminal or non zero rewards)")
parser.add_argument("--compress-replay", action='store_true', help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
parser.add_argument("--normalize-weights", action='store_true', default=True, help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
parser.add_argument("--screen-capture-freq", type=int, default=12000, help="record screens for a game this often")
parser.add_argument("--save-model-freq", type=int, default=10000, help="save the model once per 10000 training sessions")
parser.add_argument("--observation-steps", type=int, default=50000, help="train only after this many stesp (=4 frames)")
parser.add_argument("--learning-rate", type=float, default=0.00025, help="learning rate (step size for optimization algo)")
parser.add_argument("--target-model-update-freq", type=int, default=10000, help="how often (in steps) to update the target model.  Note nature paper says this is in 'number of parameter updates' but their code says steps. see tinyurl.com/hokp4y8")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("--preprocess", help="Apply preprocessing from the following: stack, ema, none", type=str, default="stack")
parser.add_argument("--conv-preprocess", type=str2bool, default=False, help="Boolean toggling using a convolutional net")
parser.add_argument("--testing", type=str2bool, default=False, help="Boolean toggling basic testing")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--save-imgs", type=str2bool, default=True, help="Boolean toggling saving of images")
parser.add_argument("rom", help="rom file to run")
args = parser.parse_args()

baseOutputDir = 'game-out-' + time.strftime("%Y-%m-%d-%H-%M-%S") if not args.testing else 'test'
if os.path.exists(baseOutputDir) and baseOutputDir == 'test':
    shutil.rmtree(baseOutputDir)
    # args.observation_steps = args.batch_size
    # args.targetModelUpdateFrequency = 1000
elif os.path.exists(baseOutputDir):
    raise ValueError("Output dir already created something went wrong")
os.makedirs(baseOutputDir)

print 'Arguments: %s' % (args)

train_log = os.path.join(baseOutputDir, 'train.txt')
train_file = open(train_log, "w")
accum_frames = 0

State.setup(args)

environment = AtariEnvironment(args, baseOutputDir)

dqn = dqn.DeepQNetwork(environment.getNumActions(), baseOutputDir, args)

replayMemory = replay.ReplayMemory(args)
first = True

def runEpoch(minEpochSteps, evalWithEpsilon=None):
    global accum_frames, first
    stepStart = environment.getStepNumber()
    isTraining = True if evalWithEpsilon is None else False
    startGameNumber = environment.getGameNumber()
    epochTotalScore = 0

    while environment.getStepNumber() - stepStart < minEpochSteps:

        startTime = lastLogTime = time.time()
        stateReward = 0
        state = None

        while not environment.isGameOver():

            # Choose next action
            if evalWithEpsilon is None:
                epsilon = max(.1, 1.0 - 0.9 * environment.getStepNumber() / 1e6)
            else:
                epsilon = evalWithEpsilon

            if state is None or random.random() > (1 - epsilon):
                action = random.randrange(environment.getNumActions())
            else:
                screens = np.reshape(state.getScreens(), (1, 84, 84, 4))
                action = dqn.inference(screens)

            # Make the move
            oldState = state
            reward, state, isTerminal = environment.step(action)

            # Record experience in replay memory and train
            if isTraining and oldState is not None:
                clippedReward = min(1, max(-1, reward))
                replayMemory.addSample(replay.Sample(oldState, action, clippedReward, state, isTerminal))
                if environment.getStepNumber() > args.observation_steps and environment.getEpisodeStepNumber() % 4 == 0:
                    if first:
                        print "STARTED TRAINING"
                        first = False
                    batch = replayMemory.drawBatch(args.batch_size)
                    dqn.train(batch, environment.getStepNumber())

            if time.time() - lastLogTime > 60:
                print('  ...frame %d' % environment.getEpisodeFrameNumber())
                lastLogTime = time.time()

            if isTerminal:
                state = None

        episodeTime = time.time() - startTime
        print('%s %d ended with score: %d (%d frames in %fs for %d fps)' %
            ('Episode' if isTraining else 'Eval', environment.getGameNumber(), environment.getGameScore(),
            environment.getEpisodeFrameNumber(), episodeTime, environment.getEpisodeFrameNumber() / episodeTime))


        accum_frames += environment.getEpisodeFrameNumber()
        log_str = "type:{} gameNum:{} score:{} framesInGame:{} accumFrames:{}".format('Episode' if isTraining else 'Eval',
                                                                             environment.getGameNumber(),
                                                                             environment.getGameScore(),
                                                                             environment.getEpisodeFrameNumber(),
                                                                             accum_frames
                                                                                      )
        # print(log_str)
        train_file.write(log_str + '\n')

        # tf.summary.scalar('GameScore', environment.getGameScore())
        # sum_str = dqn.sess.run(dqn.merged)
        # dqn.summary_writer.add_summary(sum_str, environment.getGameNumber())
        # dqn.summary_writer.flush()

        epochTotalScore += environment.getGameScore()
        environment.resetGame()

    # return the average score
    return epochTotalScore / (environment.getGameNumber() - startGameNumber)


while True:
    try:
        aveScore = runEpoch(args.train_epoch_steps) #train
        print('Average training score: %d' % (aveScore))
        aveScore = runEpoch(args.eval_epoch_steps, evalWithEpsilon=.05) #eval
        print('Average eval score: %d' % (aveScore))
    except (KeyboardInterrupt) as e:
        print(e)
        dqn.sess.close()
        train_file.close()
        if args.testing:
            shutil.rmtree(baseOutputDir)
