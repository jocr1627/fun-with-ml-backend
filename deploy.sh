#!/bin/bash

rm -rf src/__pycache__
pip3 install -r requirements.txt
python3 src/main.py
