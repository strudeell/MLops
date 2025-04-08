import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    california_dataset = fetch_california_housing()
    df = pd.DataFrame(data = california_dataset.data, columns=california_dataset.feature_names)
    df['PRICE'] = california_dataset.target
    df.to_csv("houses.csv", index = False)
    return df
def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    df = df[df['PRICE']<=4.28]
    df = df[df['MedInc']<=7.32]
    df = df[(df['AveRooms']<=7.8) & (df['AveRooms']>=2.5)]
    df = df[(df['AveBedrms']<=1.2) & (df['AveBedrms']>=0.89)]
    df = df[df['Population']<=2500]
    df = df[(df['AveOccup']<=4.3) & (df['AveOccup']>=1.27)]
    
    df.to_csv('df_clear.csv')
    return True

dag_cars = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_cars", dag = dag_cars)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_cars", dag = dag_cars)
train_task = PythonOperator(python_callable=train, task_id = "train_cars", dag = dag_cars)
download_task >> clear_task >> train_task