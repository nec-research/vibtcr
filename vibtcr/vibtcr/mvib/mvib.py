#  *     vibtcr
#  *
#  *        File:  mvib.py
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
#  *     Copyright (c) 2019 Proprietor: Deleted for purposes of anonymity
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
""" Multimodal Variational Information Bottleneck and
Attentive Variational Information Bottleneck.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from vibtcr.blocks import *
from vibtcr.base.model import BaseVIB


class MVIB(BaseVIB):
    """ This class implements the
    Multimodal Variational Information Bottleneck (MVIB) and the
    Attentive Variational Information Bottleneck (AVIB) for
    TCR recognition.
    """
    def __init__(
            self,
            z_dim: int,
            device: torch.device,
            joint_posterior: str = "aoe",
            softmax: bool = False,
            layer_norm: bool = False
    ):
        """ If `joint_posterior` = `poe`, it implements MVIB.
        If `joint_posterior` = `aoe`, it implements AVIB.

        See base class for other attributes.
        """
        super(MVIB, self).__init__(z_dim, device, joint_posterior, softmax, layer_norm)

    def forward(self, pep, cdr3b=None, cdr3a=None):
        """ Forward pass.
        """
        assert pep is not None, "Peptide must be passed."
        assert (cdr3b is not None) or (cdr3a is not None), \
            "At least CDR3b or CDR3a must be passed, together with the peptide."

        # infer joint posterior
        mu, logvar = self.infer(pep, cdr3b, cdr3a)

        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # classification
        cls_logits = self.classifier(z)

        return mu, logvar, cls_logits

    def infer(self, pep, cdr3b=None, cdr3a=None):
        """ Infer joint posterior q(z|x).
        """
        # get single-modality posteriors
        mu, logvar = self.get_unimodal_posteriors(pep, cdr3b, cdr3a)

        # combine single-modality posteriors
        # in a joint multimodal posterior
        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def get_unimodal_posteriors(self, pep, cdr3b=None, cdr3a=None):
        """ Do unimodal inference and get single-modality posteriors.
        """
        batch_size = pep.size(0)

        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.z_dim))

        pep_mu, pep_logvar = self.encoder_pep(pep)
        mu = torch.cat((mu, pep_mu.unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, pep_logvar.unsqueeze(0)), dim=0)

        if cdr3b is not None:
            cdr3b_mu, cdr3b_logvar = self.encoder_cdr3b(cdr3b)
            mu = torch.cat((mu, cdr3b_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, cdr3b_logvar.unsqueeze(0)), dim=0)

        if cdr3a is not None:
            cdr3a_mu, cdr3a_logvar = self.encoder_cdr3a(cdr3a)
            mu = torch.cat((mu, cdr3a_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, cdr3a_logvar.unsqueeze(0)), dim=0)

        return mu, logvar

    def classify(self, pep, cdr3b=None, cdr3a=None):
        """ Classification - Compute p(y|x).
        """
        mu, logvar = self.infer(pep, cdr3b, cdr3a)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        cls_logits = self.classifier(z)
        if self.softmax:
            prediction = torch.softmax(cls_logits, dim=1)
        else:
            prediction = torch.sigmoid(cls_logits)
        return prediction
