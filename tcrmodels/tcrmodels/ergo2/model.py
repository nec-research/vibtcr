""" Implements ERGO II.
Source code is taken from: https://github.com/IdoSpringer/ERGO-II
Commit id: 7a25b90db54252f4691b70af34f0d4f5351941c9

Original implementation has been wrapped for usability adopting the
`tcrmodels.abstract_model` interface.
"""
from tcrmodels.abstract_model import AbstractTCRModel
from tcrmodels.ergo2.Trainer import ERGOLightning
from tcrmodels.ergo2.Loader import SignedPairsDataset, get_index_dicts
from tcrmodels.ergo2.Predict import get_train_dicts, read_input_file, read_dataframe

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from typing import List, Union


class ERGO2(AbstractTCRModel):
    def __init__(
            self,
            gpu: List[int] = [0],
            random_seed: int = 42,
            train_val_ratio: int = .2,
            tcr_encoding_model: str = 'LSTM',
            cat_encoding: str = 'embedding',
            use_alpha: bool = True,
            use_vj: bool = False,
            use_mhc: bool = False,
            use_t_type: bool = False,
            aa_embedding_dim: int = 10,
            cat_embedding_dim: int = 50,
            lstm_dim: int = 500,
            encoding_dim: int = 100,
            lr: float = 1e-3,
            wd: float = 1e-5,
            dropout: float = 1e-1
    ):

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        hparams = dict()
        hparams['gpu'] = gpu
        hparams['tcr_encoding_model'] = tcr_encoding_model
        hparams['cat_encoding'] = cat_encoding
        hparams['use_alpha'] = use_alpha
        hparams['use_vj'] = use_vj
        hparams['use_mhc'] = use_mhc
        hparams['use_t_type'] = use_t_type
        hparams['aa_embedding_dim'] = aa_embedding_dim
        hparams['cat_embedding_dim'] = cat_embedding_dim
        hparams['lstm_dim'] = lstm_dim
        hparams['encoding_dim'] = encoding_dim
        hparams['lr'] = lr
        hparams['wd'] = wd
        hparams['dropout'] = dropout
        hparams['random_seed'] = random_seed
        hparams['train_val_ratio'] = train_val_ratio

        self.hparams = hparams

        self.model = None

    def train(self, train_df: pd.DataFrame, epochs: int = 1000) -> None:
        self.model = ERGOLightning(self.hparams, train_df)
        self.model.train_val_split(
            train_df,
            self.hparams['train_val_ratio'],
            self.hparams['random_seed']
        )
        self.train_df = train_df
        #logger = TensorBoardLogger("ERGO-II_paper_logs", name="paper_models")
        early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
        trainer = Trainer(
            gpus=self.hparams['gpu'],
            #logger=logger,
            early_stop_callback=early_stop_callback,
            deterministic=True,
            max_epochs=epochs
        )
        trainer.fit(self.model)

    def test(self, test_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
        train_dicts = get_index_dicts(self.train_df.to_dict('records'))

        if isinstance(test_df, str):
            test_samples, dataframe = read_input_file(test_df)
        elif isinstance(test_df, pd.DataFrame):
            test_samples, dataframe = read_dataframe(test_df)
        else:
            raise NotImplementedError('Input shall be either a path string or a pd.DataFame.')

        test_dataset = SignedPairsDataset(test_samples, train_dicts)
        batch_size = 1000
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: test_dataset.collate(
                b, tcr_encoding=self.model.tcr_encoding_model, cat_encoding=self.model.cat_encoding
            )
        )
        outputs = []
        for batch_idx, batch in enumerate(loader):
            output = self.model.validation_step(batch, batch_idx)
            if output:
                outputs.extend(output['y_hat'].tolist())
        dataframe['prediction'] = outputs
        return dataframe
