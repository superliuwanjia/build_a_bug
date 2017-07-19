import numpy as np
import scipy
import os
import sys
import argparse

from brownian import *

sys.path.insert(0, '/home/yuzhong/build_a_bug')
from dataLoader import *

data_path = "../../hmdb51/test/space_invaders_frames"
num_frame = 500
img_size = 64
dim = (64,64)

# placeholder
inp = np.zeros(shape=(1, num_frame, img_size, img_size, 3))

# load the video and save to placeholder
for i in range(num_frame):
	img = scipy.ndimage.imread(data_path + "/frame" + str(i) + ".jpg")
	resized_img = scipy.misc.imresize(img, dim)
	inp[0][i] = resized_img

b = brownian(inp)

save_video(b, "../../hmdb51/test/space_invaders_ema/test.mp4", framerate=30)