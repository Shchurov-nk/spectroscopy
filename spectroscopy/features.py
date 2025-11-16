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
        logger.info("Calculating XX correlations...")
        self.corr_XX = self.df_X.corr().abs()
        
        logger.info("Calculating Xy correlations...")
        result = []
        for target in self.df_y.columns:
            one_corr_Xy = self.df_X.corrwith(self.df_y[target]).abs()
            one_corr_Xy.name = target
            result.append(one_corr_Xy)
        self.corr_Xy = pd.concat(result, axis=1)
        logger.info("Calculated correlations")
    
    def save_correlations(self, XX_path, Xy_path):
        self.corr_XX.to_feather(XX_path)
        self.corr_Xy.to_feather(Xy_path)
        logger.info("Saved correlations")

    def fcbf(self, level_XX, level_Xy):
        """
        Fast correlation-based feature selection (FCBF)
        """
        logger.info(f"Selecting features using FCBF...")
        corr_Xy = self.corr_Xy.copy()
        # Calculate product to use all target variables info
        corr_Xy = corr_Xy.prod(axis=1)
        mask = [False] * len(self.corr_XX)
        while np.max(corr_Xy) > level_Xy:
            i_bestXY = np.argmax(corr_Xy)
            mask[i_bestXY] = True
            corr_Xy.iloc[i_bestXY] = 0
            for i in range(self.corr_XX.shape[0]):
                redundant = self.corr_XX.iloc[i_bestXY, i] > level_XX
                if not mask[i] and corr_Xy.iloc[i] > 0 and redundant:
                    corr_Xy.iloc[i] = 0
        self.mask = pd.DataFrame(
            mask, 
            index=self.corr_XX.index, 
            columns=['Targets']
            )
        logger.info(f"Selected {self.mask['Targets'].sum()} features using FCBF")
    
    def save_mask(self, mask_path):
        self.mask.to_feather(mask_path)

    def fit(self):
        self.calculate_correlations()
        self.fcbf()

    def transform(self):
        self.df_X = self.df_X.loc[:, self.mask['Targets']]
        return self.df_X