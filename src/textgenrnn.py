"""
See original source code at https://github.com/minimaxir/textgenrnn
"""

import json
from keras import backend as K, initializers
from keras.callbacks import Callback, LearningRateScheduler
from keras.engine import InputSpec, Layer
from keras.layers import Bidirectional, concatenate, Dense, Embedding, Input, LSTM, Reshape, SpatialDropout1D
from keras.models import load_model, Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import multi_gpu_model, Sequence
import numpy as np
from pkg_resources import resource_filename
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

class AttentionWeightedAverage(Layer):
  """
  Computes a weighted average of the different channels across timesteps.
  Uses 1 parameter pr. channel to compute the attention value for
  a single timestep.
  """

  def __init__(self, return_attention=False, **kwargs):
    self.init = initializers.get('uniform')
    self.supports_masking = True
    self.return_attention = return_attention
    super(AttentionWeightedAverage, self).__init__(** kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(ndim=3)]
    assert len(input_shape) == 3

    self.W = self.add_weight(
      shape=(input_shape[2], 1),
      name='{}_W'.format(self.name),
      initializer=self.init
    )
    self.trainable_weights = [self.W]
    super(AttentionWeightedAverage, self).build(input_shape)

  def call(self, x, mask=None):
    # computes a probability distribution over the timesteps
    # uses 'max trick' for numerical stability
    # reshape is done to avoid issue with Tensorflow
    # and 1-dimensional weights
    logits = K.dot(x, self.W)
    x_shape = K.shape(x)
    logits = K.reshape(logits, (x_shape[0], x_shape[1]))
    ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

    # masked timesteps have zero weight
    if mask is not None:
      mask = K.cast(mask, K.floatx())
      ai = ai * mask
    att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
    weighted_input = x * K.expand_dims(att_weights)
    result = K.sum(weighted_input, axis=1)
    if self.return_attention:
      return [result, att_weights]
    return result

  def get_output_shape_for(self, input_shape):
    return self.compute_output_shape(input_shape)

  def compute_output_shape(self, input_shape):
    output_len = input_shape[2]
    if self.return_attention:
      return [
        (input_shape[0], output_len),
        (input_shape[0], input_shape[1])
      ]
    return (input_shape[0], output_len)

  def compute_mask(self, input, input_mask=None):
    if isinstance(input_mask, list):
      return [None] * len(input_mask)
    else:
      return None

def textgenrnn_model(
  num_classes,
  cfg,
  weights_path=None,
  dropout=0.0,
  optimizer=RMSprop(lr=4e-3, rho=0.99)
):
  '''
  Builds the model architecture for textgenrnn and
  loads the specified weights for the model.
  '''

  input = Input(shape=(cfg['max_length'],), name='input')
  embedded = Embedding(
    num_classes, cfg['dim_embeddings'],
    input_length=cfg['max_length'],
    name='embedding'
  )(input)

  if dropout > 0.0:
    embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

  rnn_layer_list = []
  
  for i in range(cfg['rnn_layers']):
    prev_layer = embedded if i is 0 else rnn_layer_list[-1]
    rnn_layer_list.append(new_rnn(cfg, i+1)(prev_layer))

  seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
  attention = AttentionWeightedAverage(name='attention')(seq_concat)
  output = Dense(num_classes, name='output', activation='softmax')(attention)

  model = Model(inputs=[input], outputs=[output])
  
  if weights_path is not None:
    model.load_weights(weights_path, by_name=True)
  
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

  return model

'''
Create a new LSTM layer per parameters. Unfortunately,
each combination of parameters must be hardcoded.
The normal LSTMs use sigmoid recurrent activations
for parity with CuDNNLSTM:
https://github.com/keras-team/keras/issues/8860
'''

def new_rnn(cfg, layer_num):
  use_cudnnlstm = K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_available_gpus()) > 0
  
  if use_cudnnlstm:
    from keras.layers import CuDNNLSTM
    if cfg['rnn_bidirectional']:
      return Bidirectional(
        CuDNNLSTM(cfg['rnn_size'], return_sequences=True),
        name='rnn_{}'.format(layer_num)
      )

    return CuDNNLSTM(
      cfg['rnn_size'],
      return_sequences=True,
      name='rnn_{}'.format(layer_num)
    )
  else:
    if cfg['rnn_bidirectional']:
      return Bidirectional(
        LSTM(
          cfg['rnn_size'],
          return_sequences=True,
          recurrent_activation='sigmoid'
        ),
        name='rnn_{}'.format(layer_num)
      )

    return LSTM(
      cfg['rnn_size'],
      return_sequences=True,
      recurrent_activation='sigmoid',
      name='rnn_{}'.format(layer_num)
    )

def textgenrnn_encode_cat(chars, vocab):
  '''
  One-hot encodes values at given chars efficiently by preallocating
  a zeros matrix.
  '''

  a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
  rows, cols = zip(*[(i, vocab.get(char, 0)) for i, char in enumerate(chars)])
  a[rows, cols] = 1

  return a

def generate_sequences_from_texts(
  strings,
  indices_list,
  textgenrnn,
  batch_size=128
):
  max_length = textgenrnn.config['max_length']
  meta_token = textgenrnn.META_TOKEN
  new_tokenizer = textgenrnn.tokenizer

  while True:
    np.random.shuffle(indices_list)

    X_batch = []
    Y_batch = []
    count_batch = 0

    for row in range(indices_list.shape[0]):
      text_index = indices_list[row, 0]
      end_index = indices_list[row, 1]

      text = strings[text_index]
      text = [meta_token] + list(text) + [meta_token]

      if end_index > max_length:
        x = text[end_index - max_length: end_index + 1]
      else:
        x = text[0: end_index + 1]

      y = text[end_index + 1]

      if y in textgenrnn.vocab:
        x = process_sequence([x], textgenrnn, new_tokenizer)
        y = textgenrnn_encode_cat([y], textgenrnn.vocab)

        X_batch.append(x)
        Y_batch.append(y)
        count_batch += 1

        if count_batch % batch_size == 0:
          X_batch = np.squeeze(np.array(X_batch))
          Y_batch = np.squeeze(np.array(Y_batch))

          yield (X_batch, Y_batch)

          X_batch = []
          Y_batch = []
          count_batch = 0

def process_sequence(X, textgenrnn, new_tokenizer):
  X = new_tokenizer.texts_to_sequences(X)
  X = sequence.pad_sequences(X, maxlen=textgenrnn.config['max_length'])

  return X

def textgenrnn_sample(preds, temperature, interactive=False, top_n=3):
  '''
  Samples predicted probabilities of the next character to allow
  for the network to show "creativity."
  '''

  preds = np.asarray(preds).astype('float64')

  if temperature is None or temperature == 0.0:
    return np.argmax(preds)

  preds = np.log(preds + K.epsilon()) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)

  if not interactive:
    index = np.argmax(probas)

    # prevent function from being able to choose 0 (placeholder)
    # choose 2nd best index from preds
    if index == 0:
      index = np.argsort(preds)[-2]
  else:
    # return list of top N chars/words
    # descending order, based on probability
    index = (-preds).argsort()[:top_n]

  return index

def textgenrnn_encode_sequence(text, vocab, maxlen):
  '''
  Encodes a text into the corresponding encoding for prediction with
  the model.
  '''

  encoded = np.array([vocab.get(x, 0) for x in text])
  return sequence.pad_sequences([encoded], maxlen=maxlen)

def textgenrnn_generate(
  model,
  vocab,
  indices_char,
  prefix=None,
  temperature=0.5,
  maxlen=40,
  meta_token='<s>',
  max_gen_length=300,
):
  '''
  Generates and returns a single text.
  '''
  collapse_char = ''

  if prefix:
    prefix_t = list(prefix)

  text = [meta_token] + prefix_t if prefix else [meta_token]
  next_char = ''

  if not isinstance(temperature, list):
    temperature = [temperature]

  if len(model.inputs) > 1:
    model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

  while next_char != meta_token and len(text) < max_gen_length:
    encoded_text = textgenrnn_encode_sequence(text[-maxlen:], vocab, maxlen)
    next_temperature = temperature[(len(text) - 1) % len(temperature)]
    # auto-generate text without user intervention
    next_index = textgenrnn_sample(
      model.predict(encoded_text, batch_size=1)[0],
      next_temperature
    )
    next_char = indices_char[next_index]
    text += [next_char]

  text = text[1:-1]
  text_joined = collapse_char.join(text)

  return text_joined

class YieldProgress(Callback):
  def __init__(self, update):
    self.batch = 0
    self.epoch = 0
    self.update = update

  def on_batch_end(self, batch, logs):
    self.batch = batch

    if self.update:
      self.update(self.batch, self.epoch, logs)

  def on_epoch_begin(self, epoch, logs):
    self.epoch = epoch

  def on_epoch_end(self, epoch, logs):
    if self.update:
      self.update(self.batch, self.epoch, logs)

class SaveModelWeights(Callback):
  def __init__(self, textgenrnn):
    self.textgenrnn = textgenrnn
    self.weights_name = textgenrnn.config['name']

  def on_epoch_end(self, epoch, logs):
    print('saving', "weights/{}_weights.hdf5".format(self.weights_name))
    self.textgenrnn.model.save_weights("weights/{}_weights.hdf5".format(self.weights_name))

class textgenrnn:
  META_TOKEN = '<s>'
  config = {
    'rnn_layers': 2,
    'rnn_size': 128,
    'rnn_bidirectional': False,
    'max_length': 40,
    'dim_embeddings': 100,
  }

  def __init__(
    self,
    name,
    weights_path=None,
    vocab_path=None,
    config_path=None,
  ):
    if weights_path is None:
      weights_path = resource_filename(__name__, 'default_weights.hdf5')

    if vocab_path is None:
      vocab_path = resource_filename(__name__, 'vocab.json')

    if config_path is not None:
      with open(config_path, 'r', encoding='utf8', errors='ignore') as json_file:
        self.config = json.load(json_file)

    self.config.update({'name': name})

    with open(
      vocab_path,
      'r',
      encoding='utf8',
      errors='ignore'
    ) as json_file:
      self.vocab = json.load(json_file)

    self.tokenizer = Tokenizer(filters='', char_level=True)
    self.tokenizer.word_index = self.vocab
    self.num_classes = len(self.vocab) + 1
    self.model = textgenrnn_model(
      self.num_classes,
      cfg=self.config,
      weights_path=weights_path
    )
    self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

  def generate(self, max_gen_length=1, prefix=None, temperature=0.5):
    return textgenrnn_generate(
        self.model,
        self.vocab,
        self.indices_char,
        prefix,
        temperature,
        self.config['max_length'],
        self.META_TOKEN,
        max_gen_length
      )

  def train(
    self,
    strings,
    batch_size=128,
    num_epochs=1,
    train_size=1.0,
    update=None
  ):
    indices_list = [
      np.meshgrid(
        np.array(i),
        np.arange(len(string) + 1)
      ) for i,string in enumerate(strings)
    ]
    indices_list = np.block(indices_list)
    indices_mask = np.random.rand(indices_list.shape[0]) < train_size
    indices_list = indices_list[indices_mask, :]
    num_tokens = indices_list.shape[0]

    assert num_tokens >= batch_size, "Fewer tokens than batch_size."

    steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)
    gen = generate_sequences_from_texts(strings, indices_list, self, batch_size)

    def lr_linear_decay(epoch):
        return (4e-3 * (1 - (epoch / num_epochs)))

    self.model.fit_generator(
      gen,
      callbacks=[
        LearningRateScheduler(lr_linear_decay),
        YieldProgress(update),
        SaveModelWeights(self)
      ],
      epochs=num_epochs,
      max_queue_size=10,
      steps_per_epoch=steps_per_epoch
    )
