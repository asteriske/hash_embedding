import tensorflow as tf 
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras import Model

from googw2v import config
conf = config.load()

class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=conf['num_ns']+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

def build_model():
    model = Word2Vec(vocab_size=conf['vocab_size'], embedding_dim=conf['embedding_dim'])
    embedding_dim = conf['embedding_dim']

    word2vec = Word2Vec(conf['vocab_size'], embedding_dim)

    word2vec.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return word2vec