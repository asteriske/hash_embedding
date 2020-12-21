import argparse
import tensorflow as tf
from my_w2v import my_featurize, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action='store', required=True)
    args = parser.parse_args()

    text_dataset = my_featurize.file_to_dataset(args.file)
    # text_dataset = my_featurize.vectorize_to_unbatch(args.file)

    # for i, elem in enumerate(text_dataset.take(3)):
        # print(elem), num_parallel_calls=4   

    w2v = model.build_model()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/myw2v", profile_batch="500, 515")

    w2v.fit(text_dataset, epochs=20, callbacks=[tensorboard_callback])
