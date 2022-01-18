#  *     vibtcr
#  *
#  *        File:  trainer.py
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
""" Trainer base class.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from vibtcr.base.model import BaseVIB
from tqdm import trange

import os
import torch
import torch.optim as optim


class BaseTrainer:
    """ A class to automate training.
    It stores the weights of the model at the epoch where a validation score is maximized.
    """
    def __init__(
            self, model: BaseVIB, epochs: int, lr: float, beta: float,
            checkpoint_dir: str, lr_scheduler_param: int = 10, loss: str = "bce"
    ):
        """
        :param model: Torch model object
        :param epochs: max training epochs
        :param lr: learning rate
        :param beta: multiplier for KL divergence of joint posterior vs prior
        :param checkpoint_dir: directory for saving model checkpoints
        :param lr_scheduler_param: lr scheduler tuning
        :param loss: whether to use binary cross entropy for classification or MSE for regression
        """
        self.model = model
        self.epochs = epochs
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir

        self.val_loss = None
        self.kl_prior_loss = None
        self.supervised_loss = None
        self.kl_lupi_loss = None
        self.val_roc_auc = None

        if loss == "bce":
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
            self.regression = False
        elif loss == "mse":
            self.loss = torch.nn.MSELoss(reduction='mean')
            self.regression = True
        else:
            raise NotImplementedError("Losses allowed are `bce` or `mse`.")

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=lr_scheduler_param, factor=0.5)
        self.scheduler = CosineAnnealingLR(self.optimizer, lr_scheduler_param)

    def train(
            self,
            train_loader,
            val_loader,
            early_stopper_patience: int = 5,
            monitor: str = 'auROC'
    ):
        """ The main training loop.

        :param train_loader: the training Torch data loader
        :param val_loader: the validation Torch data loader
        :param early_stopper_patience: early stopping patience
        :param monitor: whether we consider loss or auROC for early stopping
        """
        best_val_score = float('inf')
        state = None
        early_stopper_counter = 0

        t = trange(1, self.epochs + 1)
        for epoch in t:
            self.train_step(train_loader)
            self.evaluate(val_loader)

            #self.scheduler.step(val_loss)

            if monitor == 'loss':
                assert self.val_loss is not None
                current_val_score = self.val_loss
            elif monitor == 'auROC':
                assert self.val_roc_auc is not None
                current_val_score = - self.val_roc_auc
            else:
                raise NotImplementedError('`monitor` must be either `loss` or `auROC`.')

            is_best = current_val_score < best_val_score
            best_val_score = min(current_val_score, best_val_score)

            if is_best or state is None:
                early_stopper_counter = 0
                state = {
                    'state_dict': self.model.state_dict(),
                    'best_val_score': best_val_score,
                    'z_dim': self.model.z_dim,
                    'joint_posterior': self.model.joint_posterior,
                    'device': self.model.device,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'softmax': self.model.softmax,
                    'layer_norm': self.model.layer_norm
                }

                msg = self.msg(epoch, best_val_score)
                t.set_description(msg)
                t.refresh()

            elif not is_best:
                early_stopper_counter += 1

            if early_stopper_counter >= early_stopper_patience:
                break

        return state

    def KL_prior(self, mu, logvar):
        """ This method returns the KL divergence
        D_KL(q(z|x) || p(z)), when the approximate posterior q(z|x) is
        a Gaussian with mean vector `mu` and diagonal covariance structure
        with log-variance vector `logvar`,
        and the prior p(z) is also a Gaussian N(0, I).
        In this case, the KL divergence can be integrated analytically
        as shown in:
        Kingma and Welling. Auto-Encoding Variational Bayes
        ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        """
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(KLD)

    def KL_gaussians(self, mu_p, logvar_p, mu_q, logvar_q):
        """ Returns the KL divergence between two multivariate Gaussian distributions.
        We assume multivariate Gaussian distributions with diagonal covariance structure.
        """
        logvar_p = torch.clamp(logvar_p, min=-1e10, max=1e10)
        logvar_q = torch.clamp(logvar_q, min=-1e10, max=1e10)

        cov_matrix_p = torch.diag_embed(logvar_p.exp(), offset=0, dim1=-2, dim2=-1)
        cov_matrix_q = torch.diag_embed(logvar_q.exp(), offset=0, dim1=-2, dim2=-1)

        p = MultivariateNormal(mu_p, cov_matrix_p)
        q = MultivariateNormal(mu_q, cov_matrix_q)

        KLD = torch.mean(kl_divergence(p, q))

        return KLD

    @staticmethod
    def save_checkpoint(state, folder='./', filename='model_best.pth.tar'):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        print('Saving best model: epoch {}'.format(state['epoch']))
        torch.save(state, os.path.join(folder, filename))

    def forward_pass(self, pep, cdr3b, cdr3a=None):
        """ Operate forward pass with all possible modality combinations.
        """
        pass

    def _process_batch(self, batch):
        """ Process a batch.
        """
        pass

    def train_step(self, train_loader):
        """ Compute loss and do backpropagation for a single epoch.
        """
        pass

    def evaluate(self, val_loader):
        """ Evaluate model performance on validation (or test) set.
        """
        pass

    def msg(self, epoch, best_val_score) -> str:
        """ Return message to output on command line.
        """
        pass
