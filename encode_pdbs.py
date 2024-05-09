#! /usr/bin/env python3
"""
Converts structures into 3Di sequences.

echo 'd12asa_' | ./struct2states.py encoder_.pt states_.txt --pdb_dir data/pdbs --virt 270 0 2
"""

import numpy as np
import sys
import os.path
import argparse

import torch

import create_vqvae_training_data
import extract_pdb_features


def generate_dict(n):
    a = {i: str(i) for i in range(n + 1)}
    b = {i + n - 1: str(-1 * i) for i in range(1, n)}
    return {**a, **b}


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32).to(device)).detach().cpu().numpy()


def discretize(encoder, centroids, x):
    z = predict(encoder, x)
    return np.argmin(extract_pdb_features.distance_matrix(z, centroids), axis=1)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('encoder', type=str, help='a *.pt file')
    arg.add_argument('centroids', type=str, help='np.loadtxt')
    arg.add_argument('K', type=str, help='alphabet size')
    arg.add_argument('--pdb_dir', type=str, help='path to PDBs')
    arg.add_argument('--virt', type=float, nargs=3, help='virtual center')
    arg.add_argument('--invalid-state', type=str, help='for missing coords.',
                     default='X')
    arg.add_argument('--exclude-feat', type=int, help='Do not calculate feature no.',
                     default=None)
    args = arg.parse_args()

    device = torch.device("cuda")

    LETTERS = generate_dict(int(args.K))

    encoder = torch.load(args.encoder).to(device)
    centroids = np.loadtxt(args.centroids)

    for line in sys.stdin:
        fn = line.rstrip('\n')
        feat, mask = create_vqvae_training_data.encoder_features(args.pdb_dir + '/' + fn, args.virt)

        if args.exclude_feat is not None:
            fmask = np.ones(feat.shape[1], dtype=bool)
            fmask[args.exclude_feat - 1] = 0
            feat = feat[:, fmask]
        valid_states = discretize(encoder, centroids, feat[mask])

        states = np.full(len(mask), -1)
        states[mask] = valid_states

        print(os.path.basename(fn), end=' ')
        print(''.join([LETTERS[state] + '*' if state != -1 else args.invalid_state + '*' for state in states]))