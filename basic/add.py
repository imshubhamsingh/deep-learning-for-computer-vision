import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x + y
session = tf.Session()
values = {x: 50, y: 60}

result = session.run([z], values)

print(result)
