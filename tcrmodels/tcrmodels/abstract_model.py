""" An abstract class which guarantees a common interface
fore the various TCR models.
"""
from abc import ABC, abstractmethod

import pandas as pd


class AbstractTCRModel(ABC):

    @abstractmethod
    def train(self, df: pd.DataFrame, epochs: int) -> None:
        pass

    @abstractmethod
    def test(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
