import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import dagshub
import mlflow 
import mlflow.sklearn

dagshub.init(repo_owner='marioalonsoalonsoch10', repo_name='mlops-practica-icai', mlflow=True)

random_state = 42

n_estimators = 200

wine_data = pd.read_csv("winequality.csv", sep=";")

inputs = wine_data.columns[:-1]
output = 'quality'

X = wine_data[inputs]
y = wine_data[output]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

cv_scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=2,
                random_state=random_state, n_jobs=-1, class_weight="balanced",
            ),
            X, y, cv=skf, scoring="f1_macro"
        )

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 

joblib.dump(model, 'model.pkl')

mlflow.sklearn.log_model(model, "wine-model")

mlflow.log_param("n_estimators", n_estimators) 
mlflow.log_metric("f1 mean", cv_scores.mean())
mlflow.log_metric("f1 std", cv_scores.std())
mlflow.log_metric("accuracy", accuracy)

print(f"Modelo entrenado y precisión: {accuracy:.4f}") 
print(f"\nCross-validation (5-fold, F1 macro): "
      f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")