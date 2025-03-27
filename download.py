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
    
    data = data[data['PRICE']<=4.28]
    data = data[data['MedInc']<=7.32]
    data = data[(data['AveRooms']<=7.8) & (data['AveRooms']>=2.5)]
    data = data[(data['AveBedrms']<=1.2) & (data['AveBedrms']>=0.89)]
    data = data[data['Population']<=2500]
    data = data[(data['AveOccup']<=4.3) & (data['AveOccup']>=1.27)]
    
    df.to_csv('df_clear.csv')
    return True

download_data()
clear_data("houses.csv")
