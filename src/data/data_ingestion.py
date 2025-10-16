import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data() -> pd.DataFrame:
  
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['target'] = housing.target
    return data
