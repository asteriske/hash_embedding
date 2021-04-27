import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.platform import test

import hash_embedding

class HashEmbeddingTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    def test_embedding_sum(self):
        testing_utils.layer_test(
            hash_embedding.HashEmbedding,
            kwargs={'num_words':1000,
                    'num_hash_func':3,
                    'aggregation_mode':'sum',
                    'append_weight':False},
            input_shape=(3,2),
            input_dtype='int32',
            expected_output_dtype='float32'
     
        )
    @keras_parameterized.run_all_keras_modes
    def test_embedding_sum_weight(self):
        testing_utils.layer_test(
            hash_embedding.HashEmbedding,
            kwargs={'num_words':1000,
                    'num_hash_func':3,
                    'aggregation_mode':'sum',
                    'append_weight':True},
            input_shape=(3,2),
            input_dtype='int32',
            expected_output_dtype='float32'
        )

    @keras_parameterized.run_all_keras_modes
    def test_embedding_append(self):
        testing_utils.layer_test(
            hash_embedding.HashEmbedding,
            kwargs={'num_words':1000,
                    'num_hash_func':3,
                    'aggregation_mode':'append',
                    'append_weight':False},
            input_shape=(3,2),
            input_dtype='int32',
            expected_output_dtype='float32'
        )
    @keras_parameterized.run_all_keras_modes
    def test_embedding_append_weight(self):
        testing_utils.layer_test(
            hash_embedding.HashEmbedding,
            kwargs={'num_words':1000,
                    'num_hash_func':3,
                    'aggregation_mode':'append',
                    'append_weight':True},
            input_shape=(3,2),
            input_dtype='int32',
            expected_output_dtype='float32'
        )

    @keras_parameterized.run_all_keras_modes
    def test_correctness_sum(self):
        NUM_WORDS = 1000
        NUM_HASH_BUCKETS = 100
        NUM_HASH_FUNC = 2
        EMBEDDING_WIDTH = 4

        layer = hash_embedding.HashEmbedding(num_words=NUM_WORDS, 
                                  num_hash_buckets=NUM_HASH_BUCKETS,
                                  num_hash_func=NUM_HASH_FUNC,
                                  embedding_width=EMBEDDING_WIDTH)
        model = keras.models.Sequential([layer])


        layer.set_weights([
                np.ones((NUM_WORDS,NUM_HASH_FUNC)), # Importance
                np.reshape(np.arange(0,NUM_HASH_BUCKETS*EMBEDDING_WIDTH),(NUM_HASH_BUCKETS,EMBEDDING_WIDTH)), # Embedding
                np.reshape(np.arange(NUM_HASH_FUNC*NUM_WORDS) % NUM_HASH_BUCKETS,(NUM_WORDS,NUM_HASH_FUNC)) # Hash Table
        ])                  

        # Embedding looks like
        #  array([[  0,   1,   2,   3],
        #         [  4,   5,   6,   7],
        #         [  8,   9,  10,  11],
        #         [ 12,  13,  14,  15],
        #         [ 16,  17,  18,  19],
        #         [ 20,  21,  22,  23],

        expected = np.array([[[20., 22., 24., 26.]],
                             [[36., 38., 40., 42.]],
                             [[52., 54., 56., 58.]]], dtype=np.float32)

        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict(np.array([[1], [2], [3]], dtype='int32'))
        self.assertAllClose(outputs, expected)     

    @keras_parameterized.run_all_keras_modes
    def test_correctness_sum_weight(self):
        NUM_WORDS = 1000
        NUM_HASH_BUCKETS = 100
        NUM_HASH_FUNC = 2
        EMBEDDING_WIDTH = 4

        layer = hash_embedding.HashEmbedding(num_words=NUM_WORDS, 
                                  num_hash_buckets=NUM_HASH_BUCKETS,
                                  num_hash_func=NUM_HASH_FUNC,
                                  embedding_width=EMBEDDING_WIDTH,
                                  append_weight=True)
        model = keras.models.Sequential([layer])


        layer.set_weights([
                np.ones((NUM_WORDS,NUM_HASH_FUNC)), # Importance
                np.reshape(np.arange(0,NUM_HASH_BUCKETS*EMBEDDING_WIDTH),(NUM_HASH_BUCKETS,EMBEDDING_WIDTH)), # Embedding
                np.reshape(np.arange(NUM_HASH_FUNC*NUM_WORDS) % NUM_HASH_BUCKETS,(NUM_WORDS,NUM_HASH_FUNC)) # Hash Table
        ])                  

        # Embedding looks like
        #  array([[  0,   1,   2,   3],
        #         [  4,   5,   6,   7],
        #         [  8,   9,  10,  11],
        #         [ 12,  13,  14,  15],
        #         [ 16,  17,  18,  19],
        #         [ 20,  21,  22,  23],

        expected = np.array([[[20., 22., 24., 26., 1., 1.]],
                             [[36., 38., 40., 42., 1., 1.]],
                             [[52., 54., 56., 58., 1., 1.]]], dtype=np.float32)

        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict(np.array([[1], [2], [3]], dtype='int32'))
        self.assertAllClose(outputs, expected)     
    @keras_parameterized.run_all_keras_modes
    def test_correctness_append(self):
        NUM_WORDS = 1000
        NUM_HASH_BUCKETS = 100
        NUM_HASH_FUNC = 2
        EMBEDDING_WIDTH = 4

        layer = hash_embedding.HashEmbedding(num_words=NUM_WORDS, 
                                  num_hash_buckets=NUM_HASH_BUCKETS,
                                  num_hash_func=NUM_HASH_FUNC,
                                  embedding_width=EMBEDDING_WIDTH,
                                  aggregation_mode='append')

        model = keras.models.Sequential([layer])


        layer.set_weights([
                np.ones((NUM_WORDS,NUM_HASH_FUNC)), # Importance
                np.reshape(np.arange(0,NUM_HASH_BUCKETS*EMBEDDING_WIDTH),(NUM_HASH_BUCKETS,EMBEDDING_WIDTH)), # Embedding
                np.reshape(np.arange(NUM_HASH_FUNC*NUM_WORDS) % NUM_HASH_BUCKETS,(NUM_WORDS,NUM_HASH_FUNC)) # Hash Table
        ])                  

        # Embedding looks like
        #  array([[  0,   1,   2,   3],
        #         [  4,   5,   6,   7],
        #         [  8,   9,  10,  11],
        #         [ 12,  13,  14,  15],
        #         [ 16,  17,  18,  19],
        #         [ 20,  21,  22,  23],

        expected = np.array([[[8.,   9., 10., 11., 12., 13., 14., 15.]],
                             [[16., 17., 18., 19., 20., 21., 22., 23.]],
                             [[24., 25., 26., 27., 28., 29., 30., 31.]]], dtype=np.float32)

        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict(np.array([[1], [2], [3]], dtype='int32'))
        self.assertAllClose(outputs, expected)    

    @keras_parameterized.run_all_keras_modes
    def test_correctness_append_weights(self):
        NUM_WORDS = 1000
        NUM_HASH_BUCKETS = 100
        NUM_HASH_FUNC = 2
        EMBEDDING_WIDTH = 4

        layer = hash_embedding.HashEmbedding(num_words=NUM_WORDS, 
                                  num_hash_buckets=NUM_HASH_BUCKETS,
                                  num_hash_func=NUM_HASH_FUNC,
                                  embedding_width=EMBEDDING_WIDTH,
                                  aggregation_mode='append',
                                  append_weight=True)

        model = keras.models.Sequential([layer])


        layer.set_weights([
                np.ones((NUM_WORDS,NUM_HASH_FUNC)), # Importance
                np.reshape(np.arange(0,NUM_HASH_BUCKETS*EMBEDDING_WIDTH),(NUM_HASH_BUCKETS,EMBEDDING_WIDTH)), # Embedding
                np.reshape(np.arange(NUM_HASH_FUNC*NUM_WORDS) % NUM_HASH_BUCKETS,(NUM_WORDS,NUM_HASH_FUNC)) # Hash Table
        ])                  

        # Embedding looks like
        #  array([[  0,   1,   2,   3],
        #         [  4,   5,   6,   7],
        #         [  8,   9,  10,  11],
        #         [ 12,  13,  14,  15],
        #         [ 16,  17,  18,  19],
        #         [ 20,  21,  22,  23],

        expected = np.array([[[8.,   9., 10., 11., 12., 13., 14., 15., 1., 1.]],
                             [[16., 17., 18., 19., 20., 21., 22., 23., 1., 1.]],
                             [[24., 25., 26., 27., 28., 29., 30., 31., 1., 1.]]], dtype=np.float32)

        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict(np.array([[1], [2], [3]], dtype='int32'))
        self.assertAllClose(outputs, expected)     


if __name__ == '__main__':
    test.main()