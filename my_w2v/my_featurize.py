import re
import string
import tensorflow as tf 

from my_w2v import config, skipgram

conf = config.load()

def write_vocab(text_vector_layer):
    vocab = text_vector_layer.get_vocabulary()
    with open(conf['vocab_file'],'w',encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i==0: continue
            f.write(word+'\n')

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')


def file_to_dataset(file: str) -> tf.data.Dataset:

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        # standardize='lower_and_strip_punctuation',
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'])


    text_ds = (
        tf.data.TextLineDataset(file).filter(lambda x: tf.cast(tf.strings.length(x),bool))
    )

    vectorize_layer.adapt(text_ds.batch(conf['batch_size'])) #fit
    write_vocab(vectorize_layer)
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(conf['vocab_size'])
    # sampling_table=None

    skipgram_func = skipgram.Skipgram(window=conf['window_size'], 
                                      vocab_size=conf['vocab_size'], 
                                      num_negative_per_example=conf['num_ns'], 
                                      frequencies=sampling_table)

    def separate_labels(target, context, labels):

        target.set_shape([conf['batch_size']])

        # context = tf.expand_dims(context,1)
        context.set_shape([conf['batch_size'],conf['num_ns']+1,1])

        labels.set_shape([conf['batch_size'],conf['num_ns']+1])
        return (
            (target, context), labels)

    text_vector_ds = (
        text_ds
        .batch(conf['batch_size'])
        .map(vectorize_layer)
        .unbatch()
    # )
    #     .take(10)
        .map(skipgram_func)
        .unbatch()
        .shuffle(conf['buffer_size'])
        .batch(conf['batch_size'], drop_remainder=True)
        .map(separate_labels)
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    # text_vector_len = len(list(text_vector_ds.as_numpy_iterator()))
    # print("text vector len")
    # print(text_vector_len)

    # target_len_ds = (
    #     text_ds
    #     .batch(conf['batch_size'])
    #     .map(vectorize_layer)
    #     .map(skipgram_func)
    #     .unbatch()
    # )
    # target_vector_len = len(list(target_len_ds.as_numpy_iterator()))
    # print('target vector len')
    # print(target_vector_len)


    return text_vector_ds
    # skipgrammed_vector_ds = (
    #     text_vector_ds
    #     # .batch(conf['batch_size'])
    #     .map(skipgram_func)
    #     .unbatch()
    #     .batch(conf['batch_size'],drop_remainder=True)
    #     .map(separate_labels)
    #     .shuffle(conf['buffer_size'])#.batch(conf['batch_size'], drop_remainder=True)
    #     .cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # )
    

    return skipgrammed_vector_ds 