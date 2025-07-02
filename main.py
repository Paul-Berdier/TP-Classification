from sklearn.model_selection import train_test_split, GridSearchCV
from diabetes_model import DiabetesModel
import pandas as pd
import joblib

# Chargement
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Définition des modèles et grilles associées
models_and_params = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
}

# Boucle sur les deux modèles
for model_type, param_grid in models_and_params.items():
    print(f"\n🧪 Entraînement du modèle : {model_type.upper()}")

    model = DiabetesModel(model_type=model_type)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"✅ Meilleurs hyperparamètres pour {model_type}: {grid.best_params_}")
    best_model.evaluate(X_test, y_test)

    try:
        best_model.explain(X_test)
    except Exception as e:
        print(f"❌ Erreur SHAP pour {model_type} : {e}")

    joblib.dump(best_model, f"model_{model_type}.joblib")
    print(f"📦 Modèle sauvegardé sous model_{model_type}.joblib")
