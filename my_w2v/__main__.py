import argparse
import datetime
import tensorflow as tf
from my_w2v import config, my_featurize, model

conf = config.load()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action='store', required=True)
    args = parser.parse_args()

    text_dataset = my_featurize.file_to_dataset(args.file)

    # for i, elem in enumerate(text_dataset):
    #     pass
    # print('num elem in batched dataset')
    # print(i)
    # for i, elem in enumerate(text_dataset.take(3)):
    #     print(elem)

    w2v = model.build_model()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/myw2v_{datetime.datetime.now().isoformat()}",
        embeddings_freq=10,
        embeddings_metadata=conf['vocab_file'])

    w2v.fit(text_dataset, epochs=50, callbacks=[tensorboard_callback])