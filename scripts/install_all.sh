#!/bin/bash

ROOT=$(pwd)

# detectron2
python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html

# SlowFast
cd $ROOT/SlowFast
python setup.py build develop

# fairseq
cd $ROOT/fairseq
pip install -e .

# cocoapi
cd $ROOT/cocoapi/PythonAPI
make

# coval
cd $ROOT/coval
pip install -e .
