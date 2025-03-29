import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
import os

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv", index_col=0)
    df.fillna(0, inplace=True)
    
    prices = df['PRICE']
    features = df.drop('PRICE', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'max_iter': [800, 900, 1000, 1100],
        'tol': [0.0001, 0.001, 0.0004],
        'fit_intercept': [False, True]
    }
    
    mlflow.set_experiment('models for houses')

    with mlflow.start_run():
        lr = Ridge(random_state=42)
        clf = GridSearchCV(lr, params, cv=5, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        
        # Проверка что модель обучена
        print(f"Model fitted: {hasattr(best, 'coef_')}")  # Должно быть True
        
        y_pred = best.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        
        mlflow.log_params(best.get_params())
        mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
        
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)  # Сохраняем best, а не lr!

        # Записываем путь к лучшей модели
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        with open("best_model.txt", "w") as f:
            f.write(model_uri)
        
        print(f"Model saved to: {model_uri}")
