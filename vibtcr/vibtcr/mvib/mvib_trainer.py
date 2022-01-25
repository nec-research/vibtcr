# *
#  *     vibtcr
#  *
#  *        File:  mvib_trainner.py
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
""" Trainer class for models that belong to the MVIB class.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sklearn.metrics import roc_auc_score
from vibtcr.mvib.mvib import MVIB
from vibtcr.base.trainer import BaseTrainer
from vibtcr.utils import AverageMeter

import torch
import numpy as np


class TrainerMVIB(BaseTrainer):
    """ A class to automate training.
    It stores the weights of the model at the epoch where a validation score is maximized.
    """
    def __init__(
            self, model: MVIB, epochs: int, lr: float, beta: float,
            checkpoint_dir: str, mode: str, lr_scheduler_param: int = 10,
            loss: str = "bce"
    ):
        """
        :param model: Torch model object
        :param epochs: max training epochs
        :param lr: learning rate
        :param beta: multiplier for KL divergence of joint posterior vs prior
        :param checkpoint_dir: directory for saving model checkpoints
        :param mode: `bimodal` or `trimodal`
        :param lr_scheduler_param: lr scheduler tuning
        :param loss: whether to use binary cross entropy (classification), or MSE (regression)
        """
        self.mode = mode
        super(TrainerMVIB, self).__init__(
            model, epochs, lr, beta, checkpoint_dir, lr_scheduler_param, loss
        )

    def _process_batch(self, batch):
        """ Process a batch.
        """
        if self.mode == "trimodal":
            pep, cdr3b, gt, cdr3a = batch
        elif self.mode == "bimodal":
            pep, cdr3b, gt = batch
            cdr3a = None
        else:
            raise NotImplementedError("`mode` must be `bimodal`, `trimodal`.")
        return pep, cdr3b, gt, cdr3a

    def train_step(self, train_loader):
        """ Compute loss and do backpropagation for a single epoch.
        """
        self.model.train()

        for batch_idx, batch in enumerate(train_loader):
            pep, cdr3b, gt, cdr3a = self._process_batch(batch)
            loss = 0

            # refresh the optimizer
            self.optimizer.zero_grad()
            self.model.zero_grad()

            # first forward pass
            ret = self.forward_pass(pep, cdr3b, cdr3a)
            mus, logvars, cls_logits = ret

            # compute KL divergence for each data combination
            # D_KL(q(z|x) || p(z)) between posterior and prior
            for i in range(len(mus)):
                loss += self.beta * self.KL_prior(mus[i], logvars[i])

            # compute binary cross entropy for each data combination
            for i in range(len(mus)):
                loss += self.loss(cls_logits[i], gt)
            
            # compute gradients and take step
            loss.backward()
            self.optimizer.step()

    def evaluate(self, val_loader):
        """ Evaluate model performance on validation (or test) set.
        """
        self.model.eval()
        kl_prior_loss_meter = AverageMeter()
        supervised_loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()

        gt_stack = []
        prediction_stack = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                pep, cdr3b, gt, cdr3a = self._process_batch(batch)
                batch_size = len(pep)
                loss = 0

                # forward pass
                ret = self.forward_pass(pep, cdr3b, cdr3a)
                mus, logvars, cls_logits = ret

                for i in range(len(mus)):
                    kl_prior_loss = self.beta * self.KL_prior(mus[i], logvars[i])
                    loss += kl_prior_loss

                for i in range(len(mus)):
                    supervised_loss = self.loss(cls_logits[i], gt)
                    loss += supervised_loss

                val_loss_meter.update(loss.item(), batch_size)
                kl_prior_loss_meter.update(kl_prior_loss.item(), batch_size)
                supervised_loss_meter.update(supervised_loss.item(), batch_size)

                if self.mode == "bimodal":
                    roc_auc_target = 0
                elif self.mode == "trimodal":
                    roc_auc_target = 2

                gt_stack.append(gt.cpu().numpy().squeeze().astype('int'))
                if self.model.softmax:
                    prediction_stack.append(torch.softmax(cls_logits[roc_auc_target], dim=1).detach().cpu().numpy().squeeze())
                else:
                    prediction_stack.append(torch.sigmoid(cls_logits[roc_auc_target]).detach().cpu().numpy().squeeze())

        gt_stack = np.concatenate(gt_stack)
        prediction_stack = np.concatenate(prediction_stack)
        val_roc_auc = roc_auc_score(gt_stack, np.nan_to_num(prediction_stack))

        self.val_loss = val_loss_meter.avg
        self.kl_prior_loss = kl_prior_loss_meter.avg
        self.supervised_loss = supervised_loss_meter.avg
        self.val_roc_auc = val_roc_auc

    def forward_pass(self, pep, cdr3b, cdr3a=None):
        """ Operate forward pass with all possible modality combinations.
        """
        # We can assume that peptide and CDR3b are never None in our setting
        mus, logvars, cls_logits = [], [], []

        mu_1, logvar_1, cls_logits_1 = self.model(pep=pep, cdr3b=cdr3b)
        if self.regression:
            cls_logits_1 = torch.sigmoid(cls_logits_1)
        mus.append(mu_1)
        logvars.append(logvar_1)
        cls_logits.append(cls_logits_1)

        if cdr3a is not None:
            mu_2, logvar_2, cls_logits_2 = self.model(pep=pep, cdr3a=cdr3a)
            if self.regression:
                cls_logits_2 = torch.sigmoid(cls_logits_2)
            mus.append(mu_2)
            logvars.append(logvar_2)
            cls_logits.append(cls_logits_2)

            ret = self.model(pep=pep, cdr3b=cdr3b, cdr3a=cdr3a)
            mu_3, logvar_3, cls_logits_3 = ret
            if self.regression:
                cls_logits_3 = torch.sigmoid(cls_logits_3)
            mus.append(mu_3)
            logvars.append(logvar_3)
            cls_logits.append(cls_logits_3)

        return mus, logvars, cls_logits

    def msg(self, epoch, best_val_score):
        if self.regression:
            msg = "[VAL] Best epoch {} | Best val score {:.6f} | DKL-prior {:.6f} | MSE {:.6f} |".format(
                epoch, best_val_score, self.kl_prior_loss, self.supervised_loss
            )
        else:
            msg = "[VAL] Best epoch {} | Best val score {:.6f} | DKL-prior {:.6f} | BCE {:.6f} | auROC {:.4f}".format(
                epoch, best_val_score, self.kl_prior_loss, self.supervised_loss, self.val_roc_auc
            )
        return msg
