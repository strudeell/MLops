from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
from pathlib import Path

def scale_frame(frame):
    """Масштабирование данных с возвратом трансформеров"""
    df = frame.copy()
    X, y = df.drop(columns=['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X)
    y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, y_scale, scaler, power_trans

def eval_metrics(actual, pred):
    """Вычисление метрик качества"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    # Загрузка и подготовка данных
    data_path = Path("./df_clear.csv")
    df = pd.read_csv(data_path)
    
    X, y, scaler, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Параметры для GridSearch
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }

    # Настройка MLflow
    mlflow.set_experiment("linear model cars")
    
    with mlflow.start_run():
        # Поиск лучшей модели
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, verbose=1)
        clf.fit(X_train, y_train.ravel())
        
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        
        # Вычисление метрик
        rmse, mae, r2 = eval_metrics(
            power_trans.inverse_transform(y_val), 
            y_price_pred
        )

        # Логирование параметров
        mlflow.log_params({
            "alpha": best.alpha,
            "l1_ratio": best.l1_ratio,
            "penalty": best.penalty,
            "eta0": best.eta0,
            "loss": best.loss,
            "fit_intercept": best.fit_intercept,
            "epsilon": best.epsilon
        })

        # Логирование метрик
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        # Логирование модели
        signature = infer_signature(X_train, best.predict(X_train))
        input_example = X_train[:1]
        
        mlflow.sklearn.log_model(
            sk_model=best,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        # Сохранение дополнительных артефактов
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        joblib.dump(best, artifacts_dir/"best_model.pkl")
        joblib.dump(scaler, artifacts_dir/"scaler.pkl")
        joblib.dump(power_trans, artifacts_dir/"power_transformer.pkl")
        
        mlflow.log_artifacts(artifacts_dir)
        
        # Удаление временных файлов
        for f in artifacts_dir.glob("*"):
            f.unlink()
        artifacts_dir.rmdir()

    # Получение пути к лучшей модели
    dfruns = mlflow.search_runs()
    best_run = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]
    path2model = best_run['artifact_uri'].replace("file://", "") + "/model"
    print(f"Path to best model: {path2model}")
