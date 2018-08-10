import argparse
from server import Server

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost', type=str)
parser.add_argument('--port', default=8000, type=int)
args = parser.parse_args()
server = Server(args.host, args.port)
server.start()
