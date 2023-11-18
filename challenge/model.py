import os

import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Tuple, Union, List
from datetime import datetime


MODEL_FILENAME = "../data/model.xgb"
SCALE_POS_WEIGHT = 4.44
THRESHOLD = 0.69


class DelayModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self
    ):
        if not hasattr(self, '_model'):
            self._model = xgb.XGBClassifier(
                random_state=1,
                learning_rate=0.01,
                scale_pos_weight=SCALE_POS_WEIGHT
            )
            try:
                if os.path.exists(MODEL_FILENAME):
                    self._model.load_model(MODEL_FILENAME)
                else:
                    raise FileNotFoundError(f"Model file {MODEL_FILENAME} not found.")
            except Exception as e:
                print(f"Error loading model: {e}")

    @staticmethod
    def get_period_day(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if (date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif (date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif (
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
        ):
            return 'noche'

    @staticmethod
    def is_high_season(fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or
                (fecha >= range2_min and fecha <= range2_max) or
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        if ('Fecha-O' in data.columns):
            data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
            data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
            data['min_diff'] = data.apply(self.get_min_diff, axis=1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        # Get dummy variables for specific columns
        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        tipo_vuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')

        # Ensure expected columns are present, adding missing ones with zeros
        for col in top_10_features:
            for dummy, prefix in [(opera_dummies, 'OPERA_'), (tipo_vuelo_dummies, 'TIPOVUELO_'), (mes_dummies, 'MES_')]:
                if col not in dummy.columns:
                    dummy[col] = 0

        # Select relevant columns and concatenate them into a features DataFrame
        features = pd.concat([
            opera_dummies[[col for col in top_10_features if col.startswith('OPERA_')]],
            tipo_vuelo_dummies[[col for col in top_10_features if col.startswith('TIPOVUELO_')]],
            mes_dummies[[col for col in top_10_features if col.startswith('MES_')]]
        ], axis=1)

        if target_column:
            target = data[[target_column]]
            return features, target
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target.values.ravel())
        self._model.save_model(MODEL_FILENAME)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        proba = self._model.predict_proba(features)
        predictions = [1 if prob > THRESHOLD else 0 for prob in proba[:, 1]]
        return predictions