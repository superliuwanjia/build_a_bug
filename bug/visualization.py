import sys
import os
import numpy

sys.path.insert(0, '/home/robin/Documents/build_a_bug')

from dataLoader import *
from model import *

def main():
    viz_output_folder = "/home/robin/Documents/visualization"
    if not os.path.exists(viz_output_folder):
        os.mkdir(viz_output_folder)

    #data_loader = KTHDataLoader("/home/robin/shared/KTH", 32, (32, 32))
    data_loader = KTHDataLoader("/home/robin/shared/KTH_small", 32, (32, 32))

    print "data loaded."
    bug = Bug([1, 80, 32, 32, 3], [1,6], 2, 1, 6)
    print "model created."
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
		
        for split in range(3):
            for video, label, fn in zip(data_loader._clips[split], 
                                        data_loader._labels[split], 
                                        data_loader._fns[split]): 
                feed_dict = { bug.x : np.expand_dims(video,axis=0),
                              bug.Y : np.expand_dims(label,axis=0),
                              bug.lr : 0.000001
                            }

                outs = sess.run(bug.retina_output, feed_dict = feed_dict)
                print "max:", outs.max()
                save_video(outs[:,:,:,0:3], \
                    os.path.join(viz_output_folder, fn + "_on.avi"), (32, 32))
                save_video(outs[:,:,:,3:6], \
                    os.path.join(viz_output_folder, fn + "_off.avi"), (32, 32))

                 

if __name__ == "__main__":
	main()
