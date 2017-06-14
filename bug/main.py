import numpy
from dataLoader import *
from model import *


def main():
	data_loader = KTHDataLoader("/home/robin/shared/KTH", 32, (32, 32))
	# data_loader = KTHDataLoader("/home/robin/shared/KTH_small", 32, (32, 32))
	print "data loaded."
	print "Creating the model ..."
	bug = Bug([32, 361, 32, 32, 3], [32,6], 2, 1, 6)
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
			if counter == 2000:
				break
			feed_dict = { bug.x : batch[0],
						  bug.Y : batch[1],
						  bug.lr : 0.000001
						}

			outs = sess.run([bug.state, bug.loss, summary_op], feed_dict = feed_dict)
			
			if counter%500 == 0:
				print "loss after {} batches".format(counter),outs[1]
				writer.add_summary(outs[2], counter)
			counter += 1
			# break
	print outs[1]

if __name__ == "__main__":
	main()