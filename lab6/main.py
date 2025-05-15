from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
from typing import List, Optional
import uvicorn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели (замените путь на свой)
try:
    with open("houses.joblib", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    # raise

with open("power.joblib", 'rb') as file:
    predict2price = pickle.load(file)


app = FastAPI(title="Houses Price")

def clear_data(df):
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    df = df[df['MedInc'] <= 7.32]
    df = df[(df['AveRooms'] <= 7.8) & (df['AveRooms'] >= 2.5)]
    df = df[(df['AveBedrms'] <= 1.2) & (df['AveBedrms'] >= 0.89)]
    df = df[df['Population'] <= 2500]
    df = df[(df['AveOccup'] <= 4.3) & (df['AveOccup'] >= 1.27)]
    
    return df

def featurize(dframe) -> pd.DataFrame:
    """
    Генерация новых признаков
    """
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['AveRooms', 'AveOccup', 'AveBedrms', 'Population']
    for col in required_columns:
        if col not in dframe.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    dframe['AvgRoomsPerHousehold'] = dframe['AveRooms'] / dframe['AveOccup']
    dframe['AvgBedroomsPerHousehold'] = dframe['AveBedrms'] / dframe['AveOccup']
    dframe['PopulationPerHousehold'] = dframe['Population'] / dframe['AveOccup']
    dframe['RoomsPerBedroom'] = dframe['AveRooms'] / dframe['AveBedrms']
    return dframe

# Модель входных данных
class HouseFeatures(BaseModel):
    MedInc:float
    HouseAge:float
    AveRooms:float
    AveBedrms:float
    Population:float
    AveOccup:float
    Latitude:float
    Longitude:float
    #AvgRoomsPerHousehold:float
    #AvgBedroomsPerHousehold:float
    #PopulationPerHousehold:float
    #RoomsPerBedroom:float

@app.post("/predict", summary="Predict house price")
async def predict(house: HouseFeatures):
    """
    Предсказывает стоимость дома
    """
    try:
        columns_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population","AveOccup","Latitude", "Longitude",]
        input_data = pd.DataFrame([house.dict()])
        input_data.columns = columns_names
        featurize_df = featurize(clear_data(input_data))
        print(featurize_df)
        predict = model.predict(featurize_df)[0]
        price = predict2price.inverse_transform(predict.reshape(-1,1))
        # logger.info(f"Predicted price: {price}")
        
        return {"predicted_price": round(float(price), 2)}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)