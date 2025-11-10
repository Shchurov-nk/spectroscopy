import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class FeatureSelector:
    def __init__(self, df_X, df_y):
        self.df_X = df_X
        self.df_y = df_y
        logger.info("Initialized FeatureSelector")

    def calculate_correlations(self):
        """
        Calculate absolute correlation values between:
        - input features (XX)
        - input features and targets (Xy)
        """

        self.corr_XX = self.df_X.corr().abs()
        self.corr_Xy = self.df_X.corrwith(self.df_y).abs().to_frame()
        logger.info("Successfully calculated correlations")
        return self.corr_XX, self.corr_Xy

    def fcbf(self, level_XX, level_Xy):
        """
        Fast correlation-based feature selection (FCBF)
        """

        corr_Xy = self.corr_Xy.copy().values
        mask = [False] * len(self.corr_XX)
        while np.max(corr_Xy) > level_Xy:
            i_bestXY = np.argmax(corr_Xy)
            mask[i_bestXY] = True
            corr_Xy[i_bestXY] = 0
            for i in range(self.corr_XX.shape[0]):
                not_redundant = self.corr_XX.iloc[i_bestXY, i] > level_XX
                if not mask[i] and corr_Xy[i] > 0 and not_redundant:
                    corr_Xy[i] = 0
        return mask