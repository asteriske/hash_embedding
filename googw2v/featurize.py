import re
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tqdm
import string

from googw2v import config
conf = config.load()

AUTOTUNE = tf.data.experimental.AUTOTUNE

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=conf['seed'], 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


def prepare_vectorize_layer(path_to_file, text_ds):

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'])

    vectorize_layer.adapt(text_ds.batch(conf['batch_size']))

    return vectorize_layer


def vectorize_text(path_to_file, zipped=False):

    text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    vectorize_layer = prepare_vectorize_layer(path_to_file, text_ds)

    def vectorize_text(text):
        text = tf.expand_dims(text, -1)
        return tf.squeeze(vectorize_layer(text))

    text_vector_ds = (
        text_ds
        .batch(conf['batch_size'])
        .prefetch(AUTOTUNE)
        .map(vectorize_layer)
        .unbatch()
    )

    if zipped:
        return tf.data.Dataset.zip((text_ds, text_vector_ds))
    
    return text_vector_ds


def sequences_to_dataset(text_dataset):

    sequences = list(text_dataset.as_numpy_iterator())
    targets, contexts, labels = generate_training_data(
        sequences=sequences, 
        window_size=2, 
        num_ns=4, 
        vocab_size=conf['vocab_size'], 
        seed=conf['seed'])

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(conf['buffer_size']).batch(conf['batch_size'], drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset


def file_to_dataset(file_path):
    sequences = vectorize_text(file_path)
    dataset = sequences_to_dataset(sequences)

    return dataset