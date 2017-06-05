
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
#os.makedirs("/tmp/model")
#os.makedirs("/tmp/model-subset")

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

model_path = "/home/ko/BNN/model/model.ckpt" 


def main(_):

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Running a new session  
  print("Starting 2nd session...")
  saver = tf.train.Saver()   
  with tf.Session() as sess:  
     # Initialize variables  
     init = tf.global_variables_initializer()
     sess.run(init)  
     
     
     # Restore model weights from previously saved model  
     load_path = saver.restore(sess, model_path)  
     print("Model restored from file: %s" % save_path) 
     
     for _ in range(1000):
          batch_xs, batch_ys = mnist.train.next_batch(100)
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

     # Test trained model
     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
