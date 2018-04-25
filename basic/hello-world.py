import tensorflow as tf

hello = tf.constant("Hello, new world")
sess = tf.Session()
print(sess.run(hello))