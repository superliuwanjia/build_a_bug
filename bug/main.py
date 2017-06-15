import numpy
from dataLoader import *
from model_v2 import *


def main():
    BATCH_SIZE = 5
    data_loader = KTHDataLoader("/home/robin/shared/KTH", BATCH_SIZE, (32, 32))
    # data_loader = KTHDataLoader("/home/robin/shared/KTH_small", BATCH_SIZE, (32, 32))
    print "data loaded."
    print "Creating the model ..."
    # with tf.device("gpu:1"):
    bug = Bug([BATCH_SIZE, 80, 32, 32, 3], [BATCH_SIZE,6], 2, 1, 6)
    # bug = Bug([BATCH_SIZE, 80, 32, 32, 3], [BATCH_SIZE,6], 2, 1, 6)
    # bug = Bug([32, 361, 32, 32, 3], [32,6], 2, 1, 6)
    # bug = Bug([32, 147, 32, 32, 3], [32,6], 2, 1, 6)
    print "model created."
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("logs", sess.graph)

        # Variable summaries
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        # Grad summaries
        for grad, var in bug.grads:
            try:
                tf.summary.histogram(var.name + '/gradient', grad)
            except Exception, e:
                continue
        # Collect all summaries
        summary_op = tf.summary.merge_all()

        counter = 0
        print "initializing variables ..."
        sess.run(tf.global_variables_initializer())
        for batch in data_loader.train_generator():
            if counter == 4200:
                break
            feed_dict = { bug.x : batch[0],
                          bug.Y : batch[1],
                          bug.lr : 0.001
                        }

            outs = sess.run([bug.train_ops, bug.loss, summary_op, bug.classification_acc], feed_dict = feed_dict)
            
            # if counter%500 == 0:
            if counter%5 == 0:
                print "loss after {} batches".format(counter),outs[1],outs[-1]
                writer.add_summary(outs[2], counter)
            counter += 1
            # break
    print outs[1]

if __name__ == "__main__":
    main()