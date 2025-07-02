from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


class DiabetesModel(BaseEstimator, ClassifierMixin):
    """
    Modèle de prédiction du diabète encapsulé dans un pipeline sklearn-compatible.
    Supporte RandomForest et XGBoost, avec explicabilité via SHAP.
    """

    def __init__(self, model_type='xgboost', n_estimators=100, max_depth=3, learning_rate=0.1):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.pipeline = None  # Nécessaire pour éviter erreurs sur get_params/set_params
        self._build_pipeline()

    def _build_pipeline(self):
        """
        Initialise le pipeline de transformation + modèle
        """
        self.scaler_ = StandardScaler()
        if self.model_type == 'random_forest':
            self.model_ = RandomForestClassifier(random_state=42)
        elif self.model_type == 'xgboost':
            self.model_ = XGBClassifier(
                                            eval_metric="logloss",
                                            n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            learning_rate=self.learning_rate,
                                            random_state=42
                                        )
        else:
            raise ValueError("model_type must be 'random_forest' or 'xgboost'")

        self.pipeline = Pipeline([
            ('scaler', self.scaler_),
            ('model', self.model_)
        ])

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Prédiction de classes.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Retourne les probabilités de prédiction.
        """
        return self.pipeline.predict_proba(X)

    # def score(self, X, y):
    #     """
    #     Retourne l’accuracy du modèle.
    #     """
    #     return self.pipeline.score(X, y)

    def evaluate(self, X, y_true):
        """
        Affiche les métriques de classification.
        """
        y_pred = self.predict(X)
        print("📊 Rapport de classification :")
        print(classification_report(y_true, y_pred))
        print("🧮 Matrice de confusion :")
        print(confusion_matrix(y_true, y_pred))

    def explain(self, X):
        """
        Affiche un graphique SHAP des contributions des features.
        """
        model = self.pipeline.named_steps['model']
        explainer = shap.Explainer(model, self.pipeline.named_steps['scaler'].transform(X))
        shap_values = explainer(self.pipeline.named_steps['scaler'].transform(X))
        shap.summary_plot(shap_values, X, plot_type='bar')

    def check_bias_by_age(self, X, y_true):
        """
        Analyse des performances par groupe d'âge.
        """
        y_pred = self.predict(X)
        df = X.copy()
        df["true"] = y_true
        df["pred"] = y_pred
        df["age_group"] = pd.cut(df["Age"], bins=[20, 35, 50, 80], labels=["jeune", "moyen", "senior"])
        print("⚖️ Taux de bonnes prédictions par âge :")
        print(df.groupby("age_group")[["true", "pred"]].apply(lambda g: (g["true"] == g["pred"]).mean()))
