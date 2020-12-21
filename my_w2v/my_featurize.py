import re
import string
import tensorflow as tf 

from my_w2v import config, skipgram

conf = config.load()

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

def vectorize_to_unbatch(file: str) -> tf.data.Dataset:

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    #     standardize=custom_standardization,
        standardize='lower_and_strip_punctuation',
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'])

    text_ds = (
        tf.data.TextLineDataset(file).filter(lambda x: tf.cast(tf.strings.length(x),bool))
    )

    vectorize_layer.adapt(text_ds) #fit

    skipgram_func = skipgram.Skipgram(window=3, 
                                      vocab_size=conf['vocab_size'], 
                                      num_negative_per_example=conf['num_ns'], 
                                      frequencies=None)


    text_vector_ds = (
        text_ds
        .batch(conf['batch_size'])
        .prefetch(tf.data.experimental.AUTOTUNE)
        .map(vectorize_layer)
        .unbatch()
        .map(skipgram_func)
    )
    
    return text_vector_ds

def file_to_dataset(file: str) -> tf.data.Dataset:

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    #     standardize=custom_standardization,
        standardize='lower_and_strip_punctuation',
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'])

    text_ds = (
        tf.data.TextLineDataset(file).filter(lambda x: tf.cast(tf.strings.length(x),bool))
    )

    vectorize_layer.adapt(text_ds) #fit
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(conf['vocab_size'])


    skipgram_func = skipgram.Skipgram(window=2, vocab_size=conf['vocab_size'], num_negative_per_example=conf['num_ns'], frequencies=sampling_table)

    def separate_labels(target, context, labels):

        target.set_shape([1])

        context = tf.expand_dims(context,1)
        # context.set_shape([conf['batch_size'],conf['num_ns']+1,1])

        # labels.set_shape([conf['batch_size'],conf['num_ns']+1])
        return (
            (target, context), labels)

    text_vector_ds = (
        text_ds
        .batch(conf['batch_size'])
        .prefetch(tf.data.experimental.AUTOTUNE)
        .map(vectorize_layer)
        .unbatch()
        # equal to here
        .map(skipgram_func, num_parallel_calls=4)
        .unbatch()
        .map(separate_labels)

        # .shuffle(conf['buffer_size'])
        .batch(conf['batch_size'], drop_remainder=True)
        .cache()
        # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return text_vector_ds