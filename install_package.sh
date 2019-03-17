#!/usr/bin/env bash

git clone https://github.com/waleedka/coco 
cd coco/PythonAPI/
make
python setup.py build_ext install
cd ../../

conda install h5py

conda install matplotlib

conda install pandas

conda install dill

conda install tqdm

pip install overrides

pip install scikit-image