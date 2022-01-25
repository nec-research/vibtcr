#  *     vibtcr
#  *
#  *        File:  model.py
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
""" Base model for multimodal variational information bottleneck.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from torch.autograd import Variable
from vibtcr.blocks import *


class BaseVIB(nn.Module):
    """ A base class to inherit from.
    """
    def __init__(
            self,
            z_dim: int,
            device: torch.device,
            joint_posterior: str = "aoe",
            softmax: bool = False,
            layer_norm: bool = False
    ):
        """
        :param z_dim: latent dimension
        :param device: GPU or CPU for Torch
        :param joint_posterior: ProductOfExperts (`poe`) or AttentionOfExperts (`aoe`)
        :param softmax: if true, 2 neurons output; if false, 1 neuron output
        :param layer_norm: operate layer normalization for mu and sigma of the posteriors
        """
        super(BaseVIB, self).__init__()

        self.z_dim = z_dim
        self.device = device
        self.joint_posterior = joint_posterior
        self.softmax = softmax
        self.layer_norm = layer_norm

        self.encoder_pep = Encoder(z_dim=z_dim, layer_norm=layer_norm)
        self.encoder_cdr3b = Encoder(z_dim=z_dim, layer_norm=layer_norm)
        self.encoder_cdr3a = Encoder(z_dim=z_dim, layer_norm=layer_norm)

        if joint_posterior == "aoe":
            self.experts = AttentionOfExperts(z_dim, heads=5, layer_norm=layer_norm)
        elif joint_posterior == "poe":
            self.experts = ProductOfExperts()
        elif joint_posterior == "max_pool":
            self.experts = MaxPoolOfExperts()
        elif joint_posterior == "avg_pool":
            self.experts = AvgPoolOfExperts()

        if softmax:
            self.classifier = Classifier(z_dim, 2, layer_norm=layer_norm)
        else:
            self.classifier = Classifier(z_dim, 1, layer_norm=layer_norm)

    def reparametrize(self, mu, logvar):
        """ Reparameterization trick.
        Samples z from its posterior distribution.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def prior_expert(self, size):
        """ Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        :param size: dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        mu, logvar = mu.to(self.device), logvar.to(self.device)
        return mu, logvar

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        model = cls(
            checkpoint['z_dim'],
            device=device,
            joint_posterior=checkpoint['joint_posterior'],
            softmax=checkpoint['softmax'],
            layer_norm=checkpoint['layer_norm']
        )
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def forward(self, pep, cdr3b=None, cdr3a=None):
        """ Forward pass.
        """
        pass

    def infer(self, pep, cdr3b=None, cdr3a=None):
        """ Infer joint posterior q(z|x).
        """
        pass

    def classify(self, pep, cdr3b=None, cdr3a=None):
        """ Classification - Compute p(y|x).
        """
        pass
