import asyncio
from enum import IntEnum
import json
from model import Model
from threading import Thread
import websockets

class Status(IntEnum):
  DONE = 0
  ACTIVE = 1

async def delete(send, args, loop):
  model_id = args['model']['id']
  model = Model(model_id)
  strings = model.delete()
  await send()

async def generate(send, args, loop):
  model_id = args['model']['id']
  model = Model(model_id)
  count = args['count'] if 'count' in args else 1
  max_length = args['maxLength'] if 'maxLength' in args else 0
  prefix = args['prefix'] if 'prefix' in args else None
  temperature = args['temperature'] if 'temperature' in args else 0.5

  for i in range(count):
    text = model.generate(max_length=max_length, prefix=prefix, temperature=temperature)
    await send(results=text, status=Status.ACTIVE)
  
  await send()

async def train(send, args, loop):
  model_id = args['model']['id']
  model = Model(model_id)
  epochs = args['epochs'] if 'epochs' in args else 1
  selectors = args['selectors'] if 'selectors' in args else ['body']
  url = args['url']

  def update(batch, epoch, logs):
    loss = float(logs['loss'])
    future = asyncio.run_coroutine_threadsafe(
      send(
        results={
          'batch': batch,
          'epoch': epoch,
          'loss': loss
        },
        status=Status.ACTIVE
      ),
      loop
    )
    future.result()

  model.train(url, epochs=epochs, selectors=selectors, update=update)
  await send()

class Server:
  handlers = {
    'delete': delete,
    'generate': generate,
    'train': train
  }

  def __init__(self, host, port):
    self.host = host
    self.port = port

    self.loop = asyncio.new_event_loop()
    thread = Thread(target=lambda: self.loop.run_forever())
    thread.start()

  async def server(self, websocket, path):
    message = await websocket.recv()
    json_message = json.loads(message)
    key = json_message['key']
    handler = self.handlers[key]
    args = json_message['args']

    async def send(results=None, status=Status.DONE):
      message = json.dumps({ 'results': results, 'status': status })
      await websocket.send(message)

    await handler(send, args, self.loop)

  def start(self):
    serve = websockets.serve(self.server, self.host, self.port)
    asyncio.get_event_loop().run_until_complete(serve)
    asyncio.get_event_loop().run_forever()
