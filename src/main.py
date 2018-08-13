import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from server import Server

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0', type=str)
parser.add_argument('--port', default=8000, type=int)
args = parser.parse_args()
server = Server(args.host, args.port)
server.start()
