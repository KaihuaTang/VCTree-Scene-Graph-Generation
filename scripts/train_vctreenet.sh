#!/usr/bin/env bash

# Train VCTREE using different orderings

export CUDA_VISIBLE_DEVICES=1

if [ $1 == "0" ]; then
    echo "TRAINING VCTREE V1"
    python models/train_rels.py -m sgdet -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 1 -clip 5 \
        -p 2000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/motifnet-sgdet-mask/vgrel-30.tar\
        -save_dir checkpoints/motifnet-sampleonce-traintree-det -nepoch 50 -use_bias -use_encoded_box -use_rl_tree
elif [ $1 == "1" ]; then
    echo "TRAINING VCTREE V2"
    python models/train_rels.py -m sgdet -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 2 -clip 3 \
        -p 1000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/motifnet-sampleonce-traintree-det/vgrel-30.tar\
        -save_dir checkpoints/motifnet-sampleonce-fixtree-det -nepoch 50 -use_bias -use_encoded_box


elif [ $1 == "2" ]; then
    echo "TRAINING VCTREE V3"
    python models/train_rels.py -m sgcls -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 1 -clip 5 \
        -p 2000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet-purge/vgrel-19.tar\
        -save_dir checkpoints/motifnet-sampleonce-traintree-sg -nepoch 50 -use_bias -use_encoded_box -use_rl_tree
elif [ $1 == "3" ]; then
    echo "TRAINING VCTREE V3"
    python models/train_rels.py -m sgcls -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 5 -clip 5 \
        -p 500 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet-sampleonce-traintree-sg/vgrel-21.tar\
        -save_dir checkpoints/motifnet-sampleonce-fixtree-sg -nepoch 50 -use_bias -use_encoded_box

elif [ $1 == "4" ]; then
    echo "TRAINING VCTREE V3"
    python models/train_rels.py -m predcls -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 1 -clip 5 \
        -p 2000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet-pred-gtlabel/vgrel-18.tar\
        -save_dir checkpoints/motifnet-sampleonce-traintree-pred -nepoch 50 -use_bias -use_encoded_box -use_rl_tree
elif [ $1 == "5" ]; then
    echo "TRAINING VCTREE V3"
    python models/train_rels.py -m predcls -model motifnet -order confidence -nl_obj 1 -nl_edge 1 -b 5 -clip 5 \
        -p 2000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet-sampleonce-traintree-pred/vgrel-20.tar\
        -save_dir checkpoints/motifnet-sampleonce-fixtree-pred -nepoch 50 -use_bias -use_encoded_box

fi
