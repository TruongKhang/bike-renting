from time import time
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_test_split(df, ratio=(0.7,0.15,0.15)):
    ratio_train, ratio_valid, ratio_test = ratio
    total_hours = len(df['id_hour'].unique())
    print(total_hours)
    # create training, test, validation set
    print('Creating training, validation, test set')
    train_size = int(total_hours*ratio_train)
    valid_size = int(total_hours*ratio_valid)
    df_train = df[df['id_hour'] < train_size]
    df_valid = df[(df['id_hour'] >= train_size) & (df['id_hour'] < (train_size+valid_size))]
    df_test = df[(df['id_hour'] >= (train_size+valid_size))]
    X_train, y_train = df_train.drop(['id_hour', 'net_rate'], axis=1), df_train['net_rate']
    X_valid, y_valid = df_valid.drop(['id_hour', 'net_rate'], axis=1), df_valid['net_rate']
    X_test, y_test = df_test.drop(['id_hour', 'net_rate'], axis=1), df_test['net_rate']
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def learn_model(X_train, y_train, X_valid, y_valid):
    t1 = time()
    model = XGBRegressor(max_depth=7, n_estimators=500)
    model.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True, early_stopping_rounds=10)
    t2 = time()
    print('Total of training time: ', t2 - t1)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: ', rmse)
    return rmse

if __name__ == '__main__':
    df = pd.read_pickle('../data/matrix_features.pkl')
    train, valid, test = train_test_split(df)
    model = learn_model(train[0], train[1], valid[0], valid[1])
    evaluate(model, test[0], test[1])