#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install tqdm matplotlib scipy numpy scikit-learn