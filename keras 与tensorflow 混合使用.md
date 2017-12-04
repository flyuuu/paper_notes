---
title: keras 与tensorflow 混合使用 
tags: tensorfow, Fly, keras
grammar_cjkRuby: true
---

最近tensorflow更新了新版本，到1.4了。做了许多更新，当然重要的是增加了==tf.keras==. 毕竟keras对于模型搭建的方便大家都是有目共睹的。

喜欢keras风格的模型搭建而不喜欢tensorflow的方式。
但是个人觉得tensorflow的对于==loss function==定义的灵活性，还是非常便捷的，所以秉承着将二者的优势放在一起的想法，研究了一下如何混合的过程。

众所周知，keras搭建模型有两种方式，Sequential 和 function（？）这两种方式，而函数式搭建每一层返回的都是tensor结果，这就和tensorflow里面的对上了。所以做了如下初探：

```python


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# build module

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = tf.keras.layers.Dense(128, activation='relu')(img)
x = tf.keras.layers.Dense(128, activation='relu')(x)
prediction = tf.keras.layers.Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels))

train_optim = tf.train.AdamOptimizer().minimize(loss)

mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for _ in range(1000):
        batch_x, batch_y = mnist_data.train.next_batch(50)
        sess.run(train_optim, feed_dict={img: batch_x, labels: batch_y})

    acc_pred = tf.keras.metrics.categorical_accuracy(labels, prediction)
    pred = sess.run(acc_pred, feed_dict={labels: mnist_data.test.labels, img: mnist_data.test.images})

    print('accuracy: %.3f' % (sum(pred)/len(mnist_data.test.labels)))



```