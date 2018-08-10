from bs4 import BeautifulSoup
import functools
import os
import requests
from textgenrnn import textgenrnn

class Model:
  def __init__(self, model_id):
    name = "model_{}".format(model_id)
    self.weights_path = "weights/{}_weights.hdf5".format(name)
    
    if not os.path.isfile(self.weights_path):
      self.weights_path = None

    self.textgenrnn = textgenrnn(name, weights_path=self.weights_path)
  
  def delete(self):
    if self.weights_path:
      os.remove(self.weights_path)

  def generate(self, max_length=1, prefix=None, temperature=0.5):
    return self.textgenrnn.generate(max_length, prefix=prefix, temperature=temperature)

  def train(self, url, epochs=1, selectors=['body'], update=None):
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    texts = [node.text for selector in selectors for node in soup.select(selector)]
    self.textgenrnn.train(texts, num_epochs=epochs, update=update)
