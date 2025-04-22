import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error
from prepare_dataset import load_config

def model_test(config):
    with open(config['test']['model_path'], 'rb') as file:
            model = pickle.load(file)
    with open(config['test']['power_path'],'rb') as file:
          power = pickle.load(file)

    df = pd.read_csv(config['test']['testset_path'])
    price_true = df['PRICE']
    df = df.drop('PRICE', axis=1)

    pred = model.predict(df)
    price_pred = power.inverse_transform(pred.reshape(-1, 1))
    mae = mean_absolute_error(price_true, price_pred)
    return mae

if __name__ == '__main__':
      config = load_config('./src/config.yaml')
      mae = model_test(config)
      print('MAE = ', mae)
    