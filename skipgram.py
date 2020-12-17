import tensorflow as tf
from typing import Tuple

class Skipgram():
    """
    This function produces skipgrams for Word2Vec as defined in `https://www.tensorflow.org/tutorials/text/word2vec`.
    """

    def __init__(self, window: int,
                 vocab_size: int,
                 num_negative_per_example: int=4,
                 frequencies: tf.Tensor=None):

        self.frequencies = frequencies
        self.num_negative_per_example = num_negative_per_example
        self.vocab_size = vocab_size
        self.window = window


    def _downsample_by_table(self,
                             input: tf.Tensor, 
                             context_words: tf.Tensor) -> tf.Tensor:
        """
        Using an externally-provided probability vector, include
        words in the embedding with inverse frequency to expected
        occurrence.
        """
        
        random_draws = tf.random.uniform(input.shape)
        
        matching_frequencies = tf.squeeze(tf.gather(self.frequencies, input))
        
        probability_less_than_draw = tf.squeeze(tf.cast(matching_frequencies < random_draws, tf.int64))
        
        # mask the missing words to 0
        downsampled_sequence = tf.math.multiply(input, probability_less_than_draw)
        
        # zero out the rows reflecting missing words
        downsampled_context = tf.reshape(probability_less_than_draw, (input.shape[0], 1)) * context_words
        
        return downsampled_sequence, downsampled_context


    def _make_fat_diagonal(self, size: int) -> tf.Tensor:
        """
        Produces a 2d (size,size) tensor in which all elements are
        zero, including the diagonal, excluding the offset diagonal
        of width (window-1) which is set to 1. Put another way, on the
        ith row the ith item is the target, and (window-1) items ahead
        are marked 1, as well as (window-1) items behind.
    
        _make_fat_diagonal(size=5, window=3)
    
        <tf.Tensor: shape=(5, 5), dtype=int32, numpy=
        array([[0, 1, 1, 0, 0],
               [1, 0, 1, 1, 0],
               [1, 1, 0, 1, 1],
               [0, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]], dtype=int32)>
    
        """
        fat_ones = tf.linalg.band_part(
            tf.ones([size,size], dtype=tf.int64),
            num_lower=self.window-1,
            num_upper=self.window-1
        )
    
        return tf.linalg.set_diag(fat_ones, tf.zeros(size, dtype=tf.int64))


    def _make_positive_skipgrams(self, input: tf.Tensor) -> tf.Tensor:
        """
        tf_positive_skipgrams(sequence=tf.constant([1,2,3,4]), window=2)
    
        <tf.Tensor: shape=(6, 2), dtype=int32, numpy=
            array([[1, 2],
                   [2, 1],
                   [2, 3],
                   [3, 2],
                   [3, 4],
                   [4, 3]], dtype=int32)>
                   
        Each word is evaluated for frequency which may result in some target
        words being rejected.    
        """
    
        fat_diagonal = self._make_fat_diagonal(size=input.shape[0])
    
        context_words = tf.math.multiply(input, fat_diagonal)
    
        # Apply table of probabilities. If a word in the sequence is too common,
        # it will be removed from the sequence and the corresponding row of 
        # context words will be removed as well. We wait to do it until now
        # because we want the common words to appear in the windows of other 
        # words.
    
        if self.frequencies is not None:
            downsampled_sequence, downsampled_context_words = self._downsample_by_table(input, context_words)
        else:
            downsampled_sequence = input
            downsampled_context_words = context_words
    
        key_and_context_with_zeros = tf.stack([
            tf.repeat(downsampled_sequence, downsampled_sequence.shape[0]),
            tf.reshape(downsampled_context_words, [-1]),
           ],axis=1)
        # print('key_and_context_with_zeros')
        # print(key_and_context_with_zeros)
        # we don't want rows where the target is 0 nor the context word is 0
        nonzero_rows = tf.where(tf.math.multiply(
            key_and_context_with_zeros[:,0],
            key_and_context_with_zeros[:,1]))
        # print('nonzero_rows')
        # print(nonzero_rows)

        key_and_context = tf.squeeze(tf.gather(key_and_context_with_zeros, nonzero_rows))

        if tf.rank(key_and_context) == tf.TensorShape([1]):
            return tf.expand_dims(key_and_context,0)
        
        return key_and_context


    def _select_skipgram_negatives(self, positive_samples: tf.Tensor) -> tf.Tensor:
        """
        Apply `draw_negative_samples` to every element of a (,1)-shaped tensor
        of positive examples.
        """
        
        def draw_negative_samples(x):
            
            matrix_x = tf.reshape(tf.cast(x, tf.int64), (1,1))
            neg_samples, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=matrix_x,
                                                                        num_true=1,
                                                                        num_sampled=self.num_negative_per_example,
                                                                        unique=True,
                                                                        range_max=self.vocab_size)
            return neg_samples            
    
        return tf.map_fn(fn=draw_negative_samples, elems=positive_samples)


    def _label_mask(self, negative_skipgrams: tf.Tensor) -> tf.Tensor:
        """
        Assume that data will take the form 
    
        [+, -, -, -]
        [+, -, -, -]
        [+, -, -, -]
    
        That is, a 2d tensor in which the first column is positive. 
    
        We can easily create a mask of labels to match from only the 
        dimensions of the negative example tensor:
    
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        """
        num_rows = tf.shape(negative_skipgrams)[0]
        num_negative_cols = tf.shape(negative_skipgrams)[1]

        return(tf.concat([
            tf.ones((num_rows, 1)),
            tf.zeros((num_rows, num_negative_cols))
        ], axis=1))


    def __call__(self, input: tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
    
        positive_skipgrams = self._make_positive_skipgrams(input)
        # print("positives")
        # print(positive_skipgrams)
    
        negative_skipgrams = self._select_skipgram_negatives(positive_skipgrams[:, 1])
    
        labels = self._label_mask(negative_skipgrams)
    
        target = positive_skipgrams[:,0]
    
        features = tf.concat([positive_skipgrams[:,1:2],
                                      negative_skipgrams], axis=1)
        
        return target, features, labels
        # return positive_skipgrams, negative_skipgrams
        # return positive_skipgrams