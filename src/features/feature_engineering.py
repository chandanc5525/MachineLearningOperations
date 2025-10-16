import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features to the dataset
    """
    df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
    df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

    df['IncomexAge'] = df['MedInc'] * df['HouseAge']
    df['IncomexRooms'] = df['MedInc'] * df['AveRooms']

    df['MedInc_squared'] = df['MedInc'] ** 2
    df['HouseAge_squared'] = df['HouseAge'] ** 2

    df['Income_bin'] = pd.cut(df['MedInc'], bins=5, labels=False)
    df['Age_bin'] = pd.cut(df['HouseAge'], bins=4, labels=False)

    df['Log_MedInc'] = np.log1p(df['MedInc'])
    df['Log_Population'] = np.log1p(df['Population'])

    return df
