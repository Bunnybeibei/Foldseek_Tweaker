#!/bin/bash

PDBDIR=$1
K=$2

./create_vqvae_training_data.py "$PDBDIR" 270 0 2

$RUN \
./encode_pdbs.py tmp_$K/encoder.pt tmp_$K/states.txt $K \
--pdb_dir "$PDBDIR" --virt 270 0 2 \
< data/pdbs_train.txt > tmp_$K/seqs.csv