"""
KTH dataset can be found here: http://www.nada.kth.se/cvap/actions/
"""

import re
import os
import math
import random
import glob
import warnings

import numpy as np
import skvideo.io
import sklearn.utils
import scipy.misc

class KTHDataLoader:
    def __init__(self, data_path, batch_size, video_size, mode="first80"):
        """
        KTH dataset loader and preprocessor
        Args:
            data_path: folder path containing both 00sequence.txt file and videos
            batch_size: batch size
            video_size: video dimention (h, w)
            mode: Method to process video clips, "episodes" or "first80"
        """
        self._batch_size = batch_size
        self._video_size = video_size


        # KTH video splits 
        splits = [[11, 12, 13, 14, 15, 16, 17, 18],          # train
                  [19, 20, 21, 23, 24, 25, 1, 4],            # validation
                  [22, 2, 3, 5, 6, 7, 8, 9, 10]]             # test
        
        label_mapping = {"boxing":0,
                         "handclapping":1, 
                         "handwaving":2,
                         "jogging":3,
                         "running":4,
                         "walking":5}
        self._num_classes = len(label_mapping)

        # file containing KTH video frame clip intervals
        sequence_list = os.path.join(data_path, "00sequences.txt")
        sequences = self._read_sequence_list(sequence_list)
       
 
        # clip and labels for each split, will be converted into [np.arrays()] format
        self._clips = [[] for _ in range(3)]                        # resized videos
        self._labels = [[] for _ in range(3)]                       # labels
        self._fns = [[] for _ in range(3)]                          # file names
        # read video into np array and create label according to splits        
        for video_file in glob.glob(os.path.join(data_path, "*.avi")):
            fn = os.path.basename(video_file)
            fn = fn[0:len(fn) - 4]
            
            video = load_video(video_file, self._video_size)
            person_index = int(fn.split("_")[0][-2:len(fn.split("_")[0])])
            split = [i for i, j in enumerate(splits) if person_index in j][0]
            label = label_mapping[fn.split("_")[1]]

            # obtain clips from video
            video_key_in_sequences = "_".join(fn.split("_")[0:len(fn.split("_")) - 1])
            print video_key_in_sequences

            if mode == "episodes":
                for clip_index, clip_range in enumerate(sequences[video_key_in_sequences]):
                    self._labels[split].append(np.eye(len(label_mapping))[label]) 
                    self._clips[split].append(video[clip_range[0] - 1:clip_range[1] - 1, :, :, :])
                    self._fns[split].append(fn + "_" + str(clip_index))
            elif mode == "first80":
                self._labels[split].append(np.eye(len(label_mapping))[label]) 
                self._clips[split].append(video[0:80, :, :, :])
                self._fns[split].append(fn)   
            else:
                raise NotImplementedError("Unknown preprocess mode.")

        # maximum length for all clips, limit for padding
        self._clip_length = np.array(\
                reduce(lambda a, b: a + [elem.shape[0] for elem in b], 
                       self._clips, [])).max()      

        for split in range(3):
            for clip_index, (clip, label) in \
                    enumerate(zip(self._clips[split], self._labels[split])):
                self._clips[split][clip_index] = np.pad(clip, \
                    ((0, self._clip_length - clip.shape[0]), (0, 0), (0, 0), (0, 0)),\
                    mode="constant", constant_values=0)
            # shuffling
            shuffle_index = range(len(self._clips[split]))
            random.shuffle(shuffle_index)
            self._clips[split] = [self._clips[split][i] for i in shuffle_index]
            self._labels[split] = [self._labels[split][i] for i in shuffle_index]
            self._fns[split] = [self._fns[split][i] for i in shuffle_index]
            
            self._clips[split] = np.concatenate(\
                [np.expand_dims(i, axis=0) for i in self._clips[split]]) 
            self._labels[split] = np.concatenate(\
                [np.expand_dims(i, axis=0) for i in self._labels[split]])

        print self._clips[0].shape
        print self._labels[0].shape
        self._batch_index = [0 for _ in range(3)]

    @property
    def train_x(self):
        return self._clips[0]
    @property
    def train_y(self):
        return self._labels[0]
    @property
    def train_fn(self):
        return self._fns[0]
    @property
    def validation_x(self):
        return self._clips[1]
    @property
    def validation_y(self):
        return self._labels[1]
    @property
    def validation_fn(self):
        return self._fns[1]
    @property
    def test_x(self):
        return self._clips[2]
    @property
    def test_y(self):
        return self._labels[2]
    @property
    def test_fn(self):
        return self._fns[2]

    @property
    def batch_size(self):
        return self._batch_size
    def video_size(self):
        return self._video_size

    def train_generator(self):
        return self._generator(0)
    def validation_generator(self):
        return self._generator(1)
    def test_generator(self):
        return self._generator(2)

    def _generator(self, split):
        while True:
            cur_index = self._batch_index[split]
            self._batch_index[split] = (cur_index + self._batch_size) \
                % self._labels[split].shape[0]
            yield (np.take(self._clips[split], \
                           range(cur_index, cur_index + self._batch_size), \
                           axis=0, mode="wrap"), \
                   np.take(self._labels[split],\
                           range(cur_index, cur_index + self._batch_size), \
                           axis=0, mode="wrap"))
       

    def _read_sequence_list(self, sequence_list):
        """ parse KTH clips file """
        f = open(sequence_list).read().replace('\t', ' ').replace(',', '')
        sequences = {}

        for line in f.split("\r\n"):
            line = re.split(' +', line)
            sequences[line[0]] = []
            for clip_col in range(2, len(line)):
                clip_range = line[clip_col].split("-")
                sequences[line[0]].append((int(clip_range[0]),int(clip_range[1])))
        return sequences

def scale(m, scale=[0, 255]):
    if m.max() == m.min():
        return np.ones_like(m) * scale[1]
    else:
        return ((m - m.min()) * (scale[1] - scale[0]))/ (m.max() - m.min())


def load_video(video_path, dim):
    """ load video with given resolution"""
    videogen = skvideo.io.vreader(video_path)
    vid_data = []
    for frame in videogen:
        try:
            vid_data.append(scipy.misc.imresize(frame, dim))
        except:
            print len(vid_data)
    return np.array(vid_data)

def resize_video(video, dim):
    vid_data = []
    for frame in video:
        vid_data.append(scipy.misc.imresize(frame, dim))
    vid_data = np.array(vid_data)
    return vid_data

def save_video(video, video_path, dim=None, framerate=3):
    """ output video with given resolution"""
    vid_data = []
    writer = skvideo.io.FFmpegWriter(video_path, inputdict={'-framerate':str(framerate)})
    for frame in video:
        if not dim is None:
            frame = scipy.misc.imresize(frame, dim)
        writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    #d = KTHDataLoader("/home/robin/shared/KTH_small", 32, (32, 32))
    #for batch in d.train_generator():
    #    print batch[0].shape
    #    print batch[1].shape
    a = (np.random.random(size=(40,64,64,3))*255).astype(np.uint8)
    save_video(a, "1.mp4", framerate=1)
    save_video(a, "5.mp4", framerate=5)
