#  *     vibtcr
#  *
#  *        File:  blocks.py
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
""" Building blocks of MVIB and AVIB.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_DROPOUT = 0.3


class Encoder(nn.Module):
    """ Parametrizes q(z|x).
    :param z_dim: latent dimension
    """
    def __init__(self, z_dim, drop: int = _DROPOUT, layer_norm: bool = False):
        super(Encoder, self).__init__()
        # in_channels = 20 due to Blosum50 encoding
        self.conv_1 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=1, padding="same")
        self.conv_3 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=3, padding="same")
        self.conv_5 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=5, padding="same")
        self.conv_7 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=7, padding="same")
        self.conv_9 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=9, padding="same")

        self.fc1 = nn.Linear(z_dim*5, z_dim)
        self.fc2 = nn.Linear(z_dim*5, z_dim)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)
        self.layer_norm = layer_norm

        self.drop = nn.Dropout(p=drop)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        c1 = self.drop(torch.relu(self.pool(self.conv_1(x)).squeeze()))
        c3 = self.drop(torch.relu(self.pool(self.conv_3(x)).squeeze()))
        c5 = self.drop(torch.relu(self.pool(self.conv_5(x)).squeeze()))
        c7 = self.drop(torch.relu(self.pool(self.conv_7(x)).squeeze()))
        c9 = self.drop(torch.relu(self.pool(self.conv_9(x)).squeeze()))

        conv_out = torch.cat([c1, c3, c5, c7, c9], axis=-1)

        mu = self.drop(self.fc1(conv_out))
        logvar = self.drop(self.fc2(conv_out))

        if self.layer_norm:
            mu = self.ln1(mu)
            logvar = self.ln2(logvar)

        return mu, logvar


class AttentionEncoder(nn.Module):
    """ Parametrizes q(z|x) with attention.
    :param z_dim: latent dimension
    """
    def __init__(self, z_dim: int, drop: int = _DROPOUT):
        super(AttentionEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(20, 4, batch_first=True)
        self.conv_1 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=1, padding="same")
        self.conv_3 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=3, padding="same")
        self.conv_5 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=5, padding="same")
        self.conv_7 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=7, padding="same")
        self.conv_9 = nn.Conv1d(in_channels=20, out_channels=z_dim, kernel_size=9, padding="same")

        self.fc1 = nn.Linear(z_dim*5, z_dim)
        self.fc2 = nn.Linear(z_dim*5, z_dim)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)

        self.drop = nn.Dropout(p=drop)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.attention(x, x, x)[0]
        x = x.transpose(1, 2)

        c1 = self.drop(torch.relu(self.pool(self.conv_1(x)).squeeze()))
        c3 = self.drop(torch.relu(self.pool(self.conv_3(x)).squeeze()))
        c5 = self.drop(torch.relu(self.pool(self.conv_5(x)).squeeze()))
        c7 = self.drop(torch.relu(self.pool(self.conv_7(x)).squeeze()))
        c9 = self.drop(torch.relu(self.pool(self.conv_9(x)).squeeze()))
        conv_out = torch.cat([c1, c3, c5, c7, c9], axis=-1)

        mu = self.drop(self.ln1(self.fc1(conv_out)))
        logvar = self.drop(self.ln2(self.fc2(conv_out)))

        return mu, logvar


class ProductOfExperts(nn.Module):
    """ Compute parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def forward(self, mu, logvar, eps=1e-8):
        """
        :param mu: M x B x D for M experts, batch size B, dimension D
        :param logvar: M x B x D for M experts, batch size B, dimension D
        :param eps: constant for stability
        """
        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)

        return pd_mu, pd_logvar


class MaxPoolOfExperts(nn.Module):
    """ Compute joint posterior from single-modality posteriors
    using 1D max pooling.
    """
    def __init__(self):
        super(MaxPoolOfExperts, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, mu, logvar):
        """
        :param mu: M x B x D for M experts, batch size B, dimension D
        :param logvar: M x B x D for M experts, batch size B, dimension D
        """
        mu = mu.transpose(0, 1)
        logvar = logvar.transpose(0, 1)

        mu = self.pool(mu.transpose(1, 2)).squeeze()
        logvar = self.pool(logvar.transpose(1, 2)).squeeze()

        return mu, logvar


class AvgPoolOfExperts(nn.Module):
    """ Compute joint posterior from single-modality posteriors
    using 1D average pooling.
    """

    def __init__(self):
        super(AvgPoolOfExperts, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, mu, logvar):
        """
        :param mu: M x B x D for M experts, batch size B, dimension D
        :param logvar: M x B x D for M experts, batch size B, dimension D
        """
        mu = mu.transpose(0, 1)
        logvar = logvar.transpose(0, 1)

        mu = self.pool(mu.transpose(1, 2)).squeeze()
        logvar = self.pool(logvar.transpose(1, 2)).squeeze()

        return mu, logvar


class AttentionOfExperts(nn.Module):
    """ Compute joint posterior from single-modality posteriors
    using multi-head attention.
    """
    def __init__(self, z_dim: int, heads: int, drop: int = _DROPOUT, layer_norm: bool = False):
        super(AttentionOfExperts, self).__init__()
        self.heads = heads
        self.layer_norm = layer_norm
        self.attention_mu = nn.MultiheadAttention(z_dim, heads, batch_first=True)
        self.attention_logvar = nn.MultiheadAttention(z_dim, heads, batch_first=True)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, mu, logvar):
        """
        :param mu: M x B x D for M experts, batch size B, dimension D
        :param logvar: M x B x D for M experts, batch size B, dimension D
        """
        mu = mu.transpose(0, 1)
        logvar = logvar.transpose(0, 1)

        aoe_mu = self.attention_mu(mu, mu, mu)[0]
        aoe_logvar = self.attention_logvar(logvar, logvar, logvar)[0]

        aoe_mu = self.pool(aoe_mu.transpose(1, 2)).squeeze()
        aoe_logvar = self.pool(aoe_logvar.transpose(1, 2)).squeeze()

        if self.layer_norm:
            aoe_mu = self.ln1(aoe_mu)
            aoe_logvar = self.ln2(aoe_logvar)

        return aoe_mu, aoe_logvar

    def attention(self, mu, logvar):
        mu = mu.transpose(0, 1)
        logvar = logvar.transpose(0, 1)

        attention_mu = self.attention_mu(mu, mu, mu)[1]
        attention_logvar = self.attention_logvar(logvar, logvar, logvar)[1]

        return attention_mu, attention_logvar


class Classifier(nn.Module):
    """ Classifier/decoder.
    Models p(y|z).
    """
    def __init__(self, z_dim, out_dim, drop: int = _DROPOUT, layer_norm: bool = False):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, out_dim)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)
        self.layer_norm = layer_norm

        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        if self.layer_norm:
            h = self.drop(F.relu(self.ln1(self.fc1(x))))
            h = self.drop(F.relu(self.ln2(self.fc2(h))))
        else:
            h = self.drop(F.relu(self.fc1(x)))
            h = self.drop(F.relu(self.fc2(h)))

        h = self.fc3(h)
        return h


class LatentBridge(nn.Module):
    """ Estimates the latent posterior conditioned on
    the privileged information q(z|x*).
    """
    def __init__(self, z_dim, drop: int = _DROPOUT):
        super(LatentBridge, self).__init__()
        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, z_dim)
        self.fc3_1 = nn.Linear(z_dim, z_dim)
        self.fc3_2 = nn.Linear(z_dim, z_dim)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)

        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        h = self.drop(torch.relu(self.ln1(self.fc1(x))))
        h = self.drop(torch.relu(self.ln2(self.fc2(h))))

        mu = self.drop(self.fc3_1(h))
        logvar = self.drop(self.fc3_2(h))
        return mu, logvar


class Ensembler:
    """ A class to ensemble model predictions and latent
    representations from multiple models.
    """
    def __init__(self):
        self._mu = 0
        self._prediction = 0
        self._count = 0

    def add(self, mu, prediction):
        """ This method needs to be called to add a model to the ensemble.

        :param mu: Torch tensor of the z Gaussian mean
        :param prediction: Torch tensor with the model predictions on the classification task
        """
        self._mu += mu.cpu().detach().numpy()
        self._prediction += prediction.cpu().detach().numpy()
        self._count += 1

    def avg(self):
        """ This method averages the models in the ensemble.
        """
        self._mu = self._mu / self._count
        self._prediction = self._prediction / self._count

        return self._mu, self._prediction
