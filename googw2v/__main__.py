import argparse
import tensorflow as tf
from googw2v import featurize, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action='store', required=True)
    args = parser.parse_args()

    text_dataset = featurize.file_to_dataset(args.file)

    for i, elem in enumerate(text_dataset.take(3)):
        print(elem)

    w2v = model.build_model()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/goog",profile_batch="500,515")

    w2v.fit(text_dataset, epochs=20, callbacks=[tensorboard_callback])
