#  *     vibtcr
#  *
#  *        File:  dataset.py
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
A Dataset class for TCR data.
"""
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Union
from torch.autograd import Variable as V

import pandas as pd
import torch
import numpy as np
import vibtcr.utils as utils


class TCRDataset(Dataset):
    """ A TCR pytorch dataset.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            device: torch.device,
            scaler: StandardScaler = None,
            pep_col: str = "peptide",
            cdr3b_col: str = "cdr3b",
            cdr3a_col: Union[str, None] = None,
            gt_col: str = "sign",
            softmax: bool = False
    ):
        """
        :param df: the dataset data frame
        :param device: torch device
        :param scaler: a StandardScaler pre-fit object
        :param pep_col: column name of the peptides (cannot be None)
        :param cdr3b_col: column name of the CDR3 beta chains (cannot be None)
        :param cdr3a_col: column name of the CDR3 alpha chains (can be None)
        :param gt_col: column name of the ground truth label
        :param softmax: if true, encode the label using one-hot encoding
        """
        super(TCRDataset, self).__init__()
        self.df = df
        self.device = device
        self.scaler = scaler
        self.pep_col = pep_col
        self.cdr3_col = cdr3b_col
        self.cdr3a_col = cdr3a_col
        self.gt_col = gt_col
        self.softmax = softmax

        self.pep = utils.encode_aa_seqs(df[pep_col], utils.blosum50_20aa, self.get_max_aa_len(pep_col))
        self.cdr3b = utils.encode_aa_seqs(df[cdr3b_col], utils.blosum50_20aa, self.get_max_aa_len(cdr3b_col))
        if cdr3a_col:
            self.cdr3a = utils.encode_aa_seqs(df[cdr3a_col], utils.blosum50_20aa, self.get_max_aa_len(cdr3a_col))

        self.gt = df[gt_col].to_numpy()
        if softmax:
            gt = np.zeros((len(self.gt), 2))
            gt[np.arange(self.gt.size), self.gt] = 1
            self.gt = gt

        self._scale_features()
        self._to_torch_variable()
        # torch.nn.Conv1D expects input with shape (N,C_in,L),
        # where N is batch size, C_in is number of channels, and L is the lengths of the signal.
        # In our case. C_in is the length of the Blosum encoding and L the number of peptides.
        # Right now we have (N, L, C_in), so we transpose to (N,C_in,L).
        self._transpose_channels()

    def __getitem__(self, idx: int):
        if self.cdr3a_col:
            return self.pep[idx], self.cdr3b[idx], self.gt[idx], self.cdr3a[idx]
        else:
            return self.pep[idx], self.cdr3b[idx], self.gt[idx]

    def __len__(self) -> int:
        return len(self.pep)

    def get_max_aa_len(self, col: str) -> int:
        """ Get max amino acid sequence length in the dataset.
        """
        max_len = utils.get_max_aa_seq_len(
            self.df, [col]
        )
        return max_len

    def _scale_features(self):
        """ Applies StandardScaler to Blosum encodings.
        """
        # reshape to 2D
        pep = self.pep.reshape(-1, self.pep.shape[-1])
        cdr3b = self.cdr3b.reshape(-1, self.cdr3b.shape[-1])

        if self.cdr3a_col:
            cdr3a = self.cdr3a.reshape(-1, self.cdr3a.shape[-1])
            x = np.concatenate([pep, cdr3b, cdr3a])
        else:
            x = np.concatenate([pep, cdr3b])

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(x)

        # transform
        self.pep = self.scaler.transform(pep).reshape(self.pep.shape)
        self.cdr3b = self.scaler.transform(cdr3b).reshape(self.cdr3b.shape)
        if self.cdr3a_col:
            self.cdr3a = self.scaler.transform(cdr3a).reshape(self.cdr3a.shape)

    def _to_torch_variable(self):
        """ Turns numpy.ndarray into torch.autograd.Variable.
        """
        self.pep = V(torch.tensor(self.pep, dtype=torch.float32, device=self.device))
        self.cdr3b = V(torch.tensor(self.cdr3b, dtype=torch.float32, device=self.device))
        self.gt = V(torch.tensor(self.gt, dtype=torch.float32, device=self.device))

        if not self.softmax:
            self.gt = self.gt.unsqueeze(dim=1)

        if self.cdr3a_col:
            self.cdr3a = V(torch.tensor(self.cdr3a, dtype=torch.float32, device=self.device))

    def _transpose_channels(self):
        self.pep = self.pep.transpose(1, 2)
        self.cdr3b = self.cdr3b.transpose(1, 2)
        if self.cdr3a_col:
            self.cdr3a = self.cdr3a.transpose(1, 2)
