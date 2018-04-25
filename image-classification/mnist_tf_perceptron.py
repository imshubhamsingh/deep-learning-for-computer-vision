import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200

x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])

weight = tf.Variable(tf.random_normal([input_size, no_classes]))
bias = tf.Variable(tf.random_normal([no_classes]))

logits = tf.matmul(x_input, weight) + bias

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits)

loss_operation = tf.reduce_mean(softmax_cross_entropy)

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

for batch_no in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    train_image, train_label = mnist_batch[0], mnist_batch[1]
    _, loss_value = session.run([optimiser, loss_operation], feed_dict={
        x_input: train_image,
        y_input: train_label
    })
    print(loss_value)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_input, 1))