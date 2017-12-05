---
title: tensorflow 保存和加载模型 参数
tags: tensorfow, Fly
grammar_cjkRuby: true
---

```python
import tensorflow as tf

'''
只加载参数，不加载网络
'''


def save_model():
    W = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32, name='W')
    b = tf.Variable([[1, 2]], dtype=tf.float32, name='b')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.save(sess, './model/model.ckpt')

def load_model():
    W = tf.Variable(tf.truncated_normal(shape=(2, 2)), dtype=tf.float32, name='W')
    b = tf.Variable(tf.truncated_normal(shape=(1, 2)), dtype=tf.float32, name='b')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')

        print(sess.run(W))


if __name__ == '__main__':
    # save_model()
    load_model()

```