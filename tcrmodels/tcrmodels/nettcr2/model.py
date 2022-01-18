""" Implements NetTCR2.0, both alpha+beta, and single-chain.
Source code is taken from: https://github.com/mnielLab/NetTCR-2.0
Commit id: 8fa4b04b264f11a2701273c55af34649c0907198

Original implementation has been wrapped for usability adopting the
`tcrmodels.abstract_model` interface.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tcrmodels.nettcr2.utils as utils
from tcrmodels.abstract_model import AbstractTCRModel


_ENCODING = utils.blosum50_20aa


class NetTCR2(AbstractTCRModel):
    def __init__(
            self,
            architecture: str,
            cdr3a_column: str = 'CDR3a',
            cdr3b_column: str = 'CDR3b',
            single_chain_column: str = 'CDR3',
            peptide_column: str = 'peptide',
            label_column: str = 'binder',
            max_pep_len: int = 9,
            max_cdr3a_len: int = 30,
            max_cdr3b_len: int = 30,
            max_cdr3_len: int = 30,

    ):
        self.architecture = architecture
        self.cdr3a_column = cdr3a_column
        self.cdr3b_column = cdr3b_column
        self.single_chain_column = single_chain_column
        self.peptide_column = peptide_column
        self.label_column = label_column
        self.max_pep_len = max_pep_len
        self.max_cdr3a_len = max_cdr3a_len
        self.max_cdr3b_len = max_cdr3b_len
        self.max_cdr3_len = max_cdr3_len

        if architecture == 'ab':
            self.model = self.nettcr_ab()
        elif architecture == 'b' or architecture == 'a':
            self.model = self.nettcr_single_chain()
        else:
            raise NotImplementedError(f"'{architecture}' not implemented")

    def reset(self):
        self.__init__(self.architecture)

    def nettcr_ab(self):
        pep_in = Input(shape=(self.max_pep_len, 20))
        cdra_in = Input(shape=(self.max_cdr3a_len, 20))
        cdrb_in = Input(shape=(self.max_cdr3b_len, 20))

        pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
        pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
        pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
        pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
        pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

        cdra_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
        cdra_pool1 = GlobalMaxPooling1D()(cdra_conv1)
        cdra_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
        cdra_pool3 = GlobalMaxPooling1D()(cdra_conv3)
        cdra_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
        cdra_pool5 = GlobalMaxPooling1D()(cdra_conv5)
        cdra_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
        cdra_pool7 = GlobalMaxPooling1D()(cdra_conv7)
        cdra_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
        cdra_pool9 = GlobalMaxPooling1D()(cdra_conv9)

        cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
        cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
        cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
        cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
        cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)

        pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
        cdra_cat = concatenate([cdra_pool1, cdra_pool3, cdra_pool5, cdra_pool7, cdra_pool9])
        cdrb_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9])

        cat = concatenate([pep_cat, cdra_cat, cdrb_cat], axis=1)

        dense = Dense(32, activation='sigmoid')(cat)

        out = Dense(1, activation='sigmoid')(dense)

        model = (Model(inputs=[cdra_in, cdrb_in, pep_in], outputs=[out]))

        return model

    def nettcr_single_chain(self):
        pep_in = Input(shape=(self.max_pep_len, 20))
        cdrb_in = Input(shape=(self.max_cdr3_len, 20))

        pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
        pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
        pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
        pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
        pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
        pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

        cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
        cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
        cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
        cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
        cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
        cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)

        pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
        cdrb_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9])

        cat = concatenate([pep_cat, cdrb_cat], axis=1)

        dense = Dense(32, activation='sigmoid')(cat)

        out = Dense(1, activation='sigmoid')(dense)

        model = (Model(inputs=[cdrb_in, pep_in], outputs=[out]))

        return model

    def _prepare_data(self, df: pd.DataFrame):
        x = []
        if self.architecture == 'ab':
            tcra = utils.enc_list_bl_max_len(df[self.cdr3a_column], _ENCODING, self.max_cdr3a_len)
            tcrb = utils.enc_list_bl_max_len(df[self.cdr3b_column], _ENCODING, self.max_cdr3b_len)
            x.append(tcra)
            x.append(tcrb)
        elif self.architecture == 'a' or self.architecture == 'b':
            tcr = utils.enc_list_bl_max_len(df[self.single_chain_column], _ENCODING, self.max_cdr3_len)
            x.append(tcr)
        else:
            raise NotImplementedError('Only supported architectures are `a`, `b`, or `ab`.')

        pep = utils.enc_list_bl_max_len(df[self.peptide_column], _ENCODING, self.max_pep_len)
        x.append(pep)

        y = np.array(df[self.label_column])

        return x, y

    def train(self, train_df: pd.DataFrame, epochs: int) -> None:
        x_train, y_train = self._prepare_data(train_df)

        early_stop = EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=10,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )

        self.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=128,
            verbose=1,
            callbacks=[early_stop]
        )

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        x_test, y_test = self._prepare_data(test_df)
        preds = self.model.predict(x_test, verbose=0)
        test_df['prediction'] = preds
        return test_df
