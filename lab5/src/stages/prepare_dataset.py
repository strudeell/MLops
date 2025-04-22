from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data)
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['PRICE', 'MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    df = df[df['PRICE'] <= 4.28]
    df = df[df['MedInc'] <= 7.32]
    df = df[(df['AveRooms'] <= 7.8) & (df['AveRooms'] >= 2.5)]
    df = df[(df['AveBedrms'] <= 1.2) & (df['AveBedrms'] >= 0.89)]
    df = df[df['Population'] <= 2500]
    df = df[(df['AveOccup'] <= 4.3) & (df['AveOccup'] >= 1.27)]
    
    df.to_csv('df_clear.csv', index=False)
    return df

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['PRICE']), df['PRICE']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scale, Y_scale, power_trans

def featurize(dframe, config) -> pd.DataFrame:
    """
    Генерация новых признаков
    """
    logger = get_logger('FEATURIZE')
    logger.info('Create features')
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['AveRooms', 'AveOccup', 'AveBedrms', 'Population']
    for col in required_columns:
        if col not in dframe.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    dframe['AvgRoomsPerHousehold'] = dframe['AveRooms'] / dframe['AveOccup']
    dframe['AvgBedroomsPerHousehold'] = dframe['AveBedrms'] / dframe['AveOccup']
    dframe['PopulationPerHousehold'] = dframe['Population'] / dframe['AveOccup']
    dframe['RoomsPerBedroom'] = dframe['AveRooms'] / dframe['AveBedrms']

    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)
    return dframe

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    df_new_featur = featurize(df_prep, config)