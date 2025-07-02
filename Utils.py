import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Classe de transformation pour le feature engineering,
    avec gestion des outliers et normalisation.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Remplacement des outliers avant fit scaler
        X_clean = self.replace_outliers_with_median(X, [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
        ])
        # Fit scaler sur les données nettoyées
        self.scaler.fit(X_clean)
        return self

    def transform(self, X):
        X = X.copy()
        # Remplacement des outliers
        X = self.replace_outliers_with_median(X, [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
        ])
        # Normalisation
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


        return X_scaled

    def replace_outliers_with_median(self, X, columns):
        """
        Remplace les valeurs aberrantes dans les colonnes spécifiées par la médiane.
        Les valeurs aberrantes sont définies comme en dehors de l'intervalle de confiance à 99%
        basé sur la moyenne et l'écart-type (distribution normale).
        """
        X = X.copy()
        for col in columns:
            if col in X.columns:
                mean = X[col].mean()
                std = X[col].std()
                median = X[col].median()
                lower_bound = mean - 2.58 * std
                upper_bound = mean + 2.58 * std
                mask_outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
                X.loc[mask_outliers, col] = median
        return X


