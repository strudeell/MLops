import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

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

download_data()
clear_data("houses.csv")
