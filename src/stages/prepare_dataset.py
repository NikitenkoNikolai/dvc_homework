import pandas as pd
import numpy as np
import yaml
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer
sys.path.append(os.getcwd())


from src.loggers import get_logger
def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def download_data(config_path):
    import kagglehub
    path = kagglehub.dataset_download(config_path)
    return path


def preprocessing_dataframe(path2data, dataset_csv):

    df = pd.read_csv(path2data+"/"+dataset_csv)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df[df['release_date'] > '1900-01-01']
    df = df[df['popularity'] <= 100.0]
    df['title_length'] = df['original_title'].str.len()

    encoder = OrdinalEncoder()
    df[['original_title']] = encoder.fit_transform(df[['original_title']])
    return df

def scale_frame(frame):
    df = frame.copy()
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns
                    if col != 'popularity']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['original_title'])
    ])

    X = df.drop(columns=['popularity'])
    y = df['popularity']

    X_processed = preprocessor.fit_transform(X)
    power_trans = PowerTransformer()
    y_processed = power_trans.fit_transform(y.values.reshape(-1, 1))

    return X_processed, y_processed, power_trans, preprocessor

def featurize(df, config) -> None:
    """

        Генерация новых признаков
    """

    
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_weekday'] = df['release_date'].dt.weekday
    df['is_weekend_release'] = df['release_weekday'].isin([5, 6]).astype(int)
    df['years_since_release'] = 2023 - df['release_year']
    df['overview_length'] = df['overview'].str.len()
    df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['rating_power'] = df['vote_average'] * np.log1p(df['vote_count'])

    df = df[['popularity', 'vote_average', 'vote_count', 'original_title',
             'release_year', 'release_month', 'release_day', 'release_weekday',
             'is_weekend_release', 'years_since_release', 'overview_length',
             'title_length', 'vote_power', 'rating_power']]


    features_path = config['featurize']['features_path']
    df.to_csv(features_path, index=False)

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_download = download_data(config['data_load']['download_dataset_csv'])
    df_prep = preprocessing_dataframe(df_download, config['data_load']['path_dataset_csv'])
    df_new_featur = featurize(df_prep, config)