import os
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import mlflow
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
from sklearn.pipeline import Pipeline
import pickle

# from src.model_scripts.plot_model import vis_weigths
from sklearn.ensemble import ExtraTreesRegressor

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(config):
    df_train = pd.read_csv(config['data_split']['trainset_path'])
    df_test  = pd.read_csv(config['data_split']['testset_path'])
    print(df_train.shape)


    X_train,y_train = df_train.drop(columns = ['popularity']).values, df_train['popularity'].values
    X_val, y_val = df_test.drop(columns = ['popularity']).values, df_test['popularity'].values
    power_trans = PowerTransformer()
    y_train = power_trans.fit_transform(y_train.reshape(-1,1))
    y_val = power_trans.transform(y_val.reshape(-1,1))
    

    
    mlflow.set_experiment("linear model movies")
    with mlflow.start_run():
        if config['train']['model_type'] == "tree":
            lr_pipe = Pipeline(steps=[('scaler',StandardScaler()),
                                  ('model', ExtraTreesRegressor(random_state=42))])
            params = {'model__n_estimators': config['train']['n_estimators'],
            }
        else:
            lr_pipe = Pipeline(steps=[('scaler',StandardScaler()),
                                  ('model', SGDRegressor(random_state=42))])
        
            params = {'model__alpha': config['train']['alpha'],
            "model__fit_intercept": [False, True],
            }
        
        clf = GridSearchCV(lr_pipe, params, cv = config['train']['cv'], n_jobs = 4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
        print(y_price_pred[:5])
        print(y_val[:5])
        (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val.reshape(-1,1)), y_price_pred)
        print("R2=",r2)
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        os.makedirs("./models", exist_ok=True)

        with open(config['train']['model_path'], "wb") as file:
            pickle.dump(best, file)

        with open(config['train']['power_path'], "wb") as file:

            pickle.dump(power_trans, file)
    if config['train']['model_type'] == "tree":
        pass