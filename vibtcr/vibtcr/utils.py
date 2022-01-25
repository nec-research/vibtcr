#  *     vibtcr
#  *
#  *        File:  utils.py
#  *
#  *     Authors: Deleted for purposes of anonymity
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  *
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software.
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2022 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
#  *
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  *
#  * For purposes of anonymity, the identity of the proprietor is not given herewith.
#  * The identity of the proprietor will be given once the review of the
#  * conference submission is completed.
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  *
"""
Utilities.
"""
import os
import json
import numpy as np
import sys
import pandas as pd

from typing import List


blosum50_20aa = {
    'A': np.array((5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)),
    'R': np.array((-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3)),
    'N': np.array((-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3)),
    'D': np.array((-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4)),
    'C': np.array((-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)),
    'Q': np.array((-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3)),
    'E': np.array((-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3)),
    'G': np.array((0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)),
    'H': np.array((-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4)),
    'I': np.array((-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4)),
    'L': np.array((-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1)),
    'K': np.array((-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3)),
    'M': np.array((-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1)),
    'F': np.array((-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1)),
    'P': np.array((-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)),
    'S': np.array((1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2)),
    'T': np.array((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)),
    'W': np.array((-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)),
    'Y': np.array((-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)),
    'V': np.array((0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5)),
    'X': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))  # unknown amino acid X
}


def encode_aa_seqs(aa_seqs, blosum, max_seq_len):
    '''
    blosum encoding of a list of amino acid sequences with padding
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences = []
    for seq in aa_seqs:
        e_seq = np.zeros((len(seq), len(blosum["A"])))
        count = 0
        for aa in seq:
            if aa in blosum:
                e_seq[count] = blosum[aa]
                count += 1
            else:
                sys.stderr.write("Unknown amino acid in peptides: " + aa + ", encoding aborted!\n")
                sys.exit(2)

        sequences.append(e_seq)

    # pad sequences:
    # max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0, n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq


def get_max_aa_seq_len(df: pd.DataFrame, cols: List[str]) -> int:
    """ Given a list of column names, get the global maximum
    length of the peptide lengths.
    """
    max_col_length = [
        df[col].str.len().max()
        for col in cols
    ]
    return max(max_col_length)


def save_latent(experiment_dir, mu, logvar, y_test, test_ids, fold):
    """ Save latent space embeddings and labels.
    """
    np.save(os.path.join(experiment_dir, 'mu_{}'.format(fold)), mu[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'logvar_{}'.format(fold)), logvar[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'label_{}'.format(fold)), y_test)


class AverageMeter(object):
    """ Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count