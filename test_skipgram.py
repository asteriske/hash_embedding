import unittest
import tensorflow as tf
from my_w2v.skipgram import Skipgram

class TestSkipgram(unittest.TestCase):

    def test_2_words_3_negative(self):
        vocab_size=1000
        this_skipgram = Skipgram(window=3, vocab_size=vocab_size, num_negative_per_example=3, frequencies=None)

        input = tf.constant([1,2,0,0,0,0], dtype=tf.int64)

        target, features, labels = this_skipgram(input)
        tf.assert_equal(target, tf.constant([[1],[2]],dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([2,4]))
        tf.assert_equal(labels.shape, tf.TensorShape([2,4]))
        tf.assert_equal(labels, tf.constant([[1,0,0,0],
                                             [1,0,0,0]], dtype=tf.float32))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size,dtype=tf.int64))

    def test_2_words_6_negative(self):
        vocab_size=1000
        this_skipgram = Skipgram(window=3, vocab_size=vocab_size, num_negative_per_example=6, frequencies=None)

        input = tf.constant([1,2,0,0,0,0], dtype=tf.int64)
        target, features, labels = this_skipgram(input)
        tf.assert_equal(target, tf.constant([1,2],dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([2,7]))
        tf.assert_equal(labels.shape, tf.TensorShape([2,7]))
        tf.assert_equal(labels, tf.constant([[1,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0]], dtype=tf.float32))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size,dtype=tf.int64))


    def test_2_words_3_negative_rand(self):

        vocab_size=1000
        this_skipgram = Skipgram(window=3, vocab_size=vocab_size, num_negative_per_example=3, 
                                 frequencies=tf.constant([0,1,0],dtype=tf.float32))

        input = tf.constant([1,2,0,0,0,0], dtype=tf.int64)

        target, features, labels = this_skipgram(input)
        tf.assert_equal(target, tf.constant([2],dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([1,4]))
        tf.assert_equal(labels.shape, tf.TensorShape([1,4]))
        tf.assert_equal(labels, tf.constant([[1,0,0,0]], dtype=tf.float32))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size, dtype=tf.int64))


    def test_2_words_6_negative_rand(self):

        vocab_size=1000
        this_skipgram = Skipgram(window=3, vocab_size=vocab_size, num_negative_per_example=6, 
                                 frequencies=tf.constant([0,1,0],dtype=tf.float32))

        input = tf.constant([1,2,0,0,0,0], dtype=tf.int64)

        target, features, labels = this_skipgram(input)
        tf.assert_equal(target, tf.constant([2],dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([1,7]))
        tf.assert_equal(labels.shape, tf.TensorShape([1,7]))
        tf.assert_equal(labels, tf.constant([[1,0,0,0,0,0,0]], dtype=tf.float32))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size, dtype=tf.int64))


    def test_window_4_negative_3(self):
        vocab_size=1000
        this_skipgram = Skipgram(window=4, vocab_size=vocab_size, num_negative_per_example=3, frequencies=None)

        input = tf.constant([1,2,3,4,5,6,7,8], dtype=tf.int64)

        target, features, labels = this_skipgram(input)
        tf.assert_equal(target, tf.constant([1,1,1,
                                             2,2,2,2,
                                             3,3,3,3,3,
                                             4,4,4,4,4,4,
                                             5,5,5,5,5,5,
                                             6,6,6,6,6,
                                             7,7,7,7,
                                             8,8,8], dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([36,4]))
        tf.assert_equal(labels.shape, tf.TensorShape([36,4]))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size, dtype=tf.int64))


    def test_window_4_negative_6(self):
        vocab_size=1000
        this_skipgram = Skipgram(window=4, vocab_size=vocab_size, num_negative_per_example=6, frequencies=None)

        input = tf.constant([1,2,3,4,5,6,7,8], dtype=tf.int64)

        target, features, labels = this_skipgram(input)
        # result = this_skipgram(input)
        # print("input")
        # print(input)
        # print("target")
        # print(target)
        # print("features")
        # print(features)
        # print('labels')
        # print(labels)
        # print(result)
        tf.assert_equal(target, tf.constant([1,1,1,
                                             2,2,2,2,
                                             3,3,3,3,3,
                                             4,4,4,4,4,4,
                                             5,5,5,5,5,5,
                                             6,6,6,6,6,
                                             7,7,7,7,
                                             8,8,8], dtype=tf.int64))
        tf.assert_equal(features.shape, tf.TensorShape([36,7]))
        tf.assert_equal(labels.shape, tf.TensorShape([36,7]))
        tf.assert_less(tf.reduce_max(features), tf.constant(vocab_size, dtype=tf.int64))