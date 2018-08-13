#!/bin/bash

rm -rf src/__pycache__
pip3 install -r requirements.txt
mkdir weights
python3 src/main.py --port $1
