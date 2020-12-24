import argparse
import datetime
import tensorflow as tf
from googw2v import featurize, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action='store', required=True)
    args = parser.parse_args()

    text_dataset = featurize.file_to_dataset(args.file)

    # dataset_len = len(list(text_dataset.unbatch().as_numpy_iterator()))
    # print('dataset len')
    # print(dataset_len)
    # for i, elem in enumerate(text_dataset):
    #     pass
    # print('num elem in batched dataset')
    # print(i)

    # for i in text_dataset.take(5):
    #     print(i)
    w2v = model.build_model()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/goog_{datetime.datetime.now().isoformat()}", profile_batch="200,220")

    w2v.fit(text_dataset, epochs=50, callbacks=[tensorboard_callback])
