import numpy as np
from os import name
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv", index_col = 0)
    df.fillna(0, inplace=True)
    '''X,Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)'''
    prices = df['PRICE']
    features = df.drop('PRICE', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
    

    #Ridge
    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
        'max_iter':[800,900,1000,1100],
        'tol':[0.0001, 0.001, 0.0004],
        'fit_intercept':[False, True]
    }
    mlflow.set_experiment('models for houses')

    with mlflow.start_run():

        lr=Ridge(random_state=42)
        clf = GridSearchCV(lr, params, cv = 5, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_test)
        y_price_pred = y_pred
        (rmse, mae, r2)  = eval_metrics(y_test, y_price_pred)
        
        alpha = best.alpha
        max_iter = best.max_iter
        tol = best.tol
        fit_intercept = best.fit_intercept
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("tol", tol)
        mlflow.log_param("fit_intercept", fit_intercept)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)

        with open("lr_cars.pkl", "wb") as file:
            joblib.dump(lr, file)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model' #путь до эксперимента с лучшей моделью
    print(path2model)
