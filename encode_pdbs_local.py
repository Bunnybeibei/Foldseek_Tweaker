#! /usr/bin/env python3
import numpy as np
import os.path
import argparse

import torch

import create_vqvae_training_data
import extract_pdb_features

import sys


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
    THETA = 270
    TAU = 0
    D = 2

    device = torch.device("cuda")
    arg = argparse.ArgumentParser()
    arg.add_argument('--encoder', type=str, help='a *.pt file', default='tmp_512/encoder.pt')
    arg.add_argument('--centroids', type=str, help='np.loadtxt', default='tmp_512/states.txt')
    arg.add_argument('--pdb_dir', type=str, help='path to PDBs', default='tmp/pdb')
    arg.add_argument('--K', type=str, help='alphabet size', default=512)
    arg.add_argument('--virt', type=float, nargs=3, help='virtual center', default=[THETA, TAU, D])
    arg.add_argument('--invalid-state', type=str, help='for missing coords.',
                     default='X')
    arg.add_argument('--exclude-feat', type=int, help='Do not calculate feature no.',
                     default=None)
    args = arg.parse_args()

    LETTERS = generate_dict(int(args.K))

    sys.stdout = open(f'tmp_{int(args.K)}/seqs2.csv', 'w')

    encoder = torch.load(args.encoder).to(device)
    centroids = np.loadtxt(args.centroids)

    with open('data/pdbs_train.txt', 'r') as f:
        for i, line in enumerate(f):
            # if i >= 10:  # Early stop for testing
            #     break
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

    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # Save in 'tmp_K/seqs2.csv'
