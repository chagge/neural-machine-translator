#!/usr/bin/env bash
dir='saved/'

rm -rf $dir'/corpora_processed/'
sudo rm -rf $dir'/words_index/'
sudo rm -rf $dir'/w2v_models/'
sudo rm -rf $dir'/nn_models/'
sudo rm -rf $dir'/results/'


sudo mkdir -p $dir'/corpora_processed/'
sudo mkdir -p $dir'/words_index/'
sudo mkdir -p $dir'/w2v_models/'
sudo mkdir -p $dir'/nn_models/'
sudo mkdir -p $dir'/results/'

scp europarl-v7.de-en.en gpuuser@csews29.cse.iitk.ac.in:/home/gpuuser/neural-machine-translator/data/train
scp europarl-v7.de-en.de gpuuser@csews29.cse.iitk.ac.in:/home/gpuuser/neural-machine-translator/data/train