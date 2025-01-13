import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import datetime

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        datetime: datetime,
        season : int ,
        holiday : int,
        workingday : int,
        weather : int,
        temp : float,
        atemp : float,
        humidity : int,
        windspeed : float):

        self.datetime = datetime

        self.season = season

        self.holiday = holiday

        self.workingday = workingday

        self.weather = weather

        self.temp = temp

        self.atemp = atemp

        self.humidity = humidity

        self.windspeed = windspeed


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "datetime": [self.datetime],
                "season": [self.season],
                "holiday": [self.holiday],
                "workingday": [self.workingday],
                "weather": [self.weather],
                "temp": [self.temp],
                "atemp": [self.atemp],
                "humidity": [self.humidity],
                "windspeed": [self.windspeed]
            }
            df = pd.DataFrame(custom_data_input_dict)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["year"] = df["datetime"].dt.year
            df["month"] = df["datetime"].dt.month
            df["day"] = df["datetime"].dt.day
            df["hour"] = df["datetime"].dt.hour
            df["day_name"] = df["datetime"].dt.day_name()
            df.drop(["datetime"],axis = 1,inplace = True)

            return df

        except Exception as e:
            raise CustomException(e, sys)
