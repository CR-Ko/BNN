import tensorflow as tf
import os
os.makedirs("/home/ko/BNN/model-basic")
os.makedirs("/home/ko/BNN/model-basic-subset")



v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.2, 0.2], name="v2")


init_op = tf.global_variables_initializer()


saver = tf.train.Saver()

with tf.Session() as sess:
    
    
    sess.run(init_op)
    
    
    ops = tf.assign(v2, [0.3, 0.3])
    sess.run(ops)
    
    print sess.run(tf.global_variables())
    save_path = saver.save(sess, "/home/ko/BNN/model-basic/model.ckpt")
