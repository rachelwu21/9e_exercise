import psycopg2
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import xgboost
import json

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].index.dayofweek
    df['quarter'] = df['date'].index.quarter
    df['month'] = df['date'].index.month
    df['year'] = df['date'].index.year
    df['dayofyear'] = df['date'].index.dayofyear
    df['dayofmonth'] = df['date'].index.day
    df['weekofyear'] = df['date'].index.isocalendar().week
    df['yesterday'] = df["sentiment_score"].shift(1)

    df.dropna(inplace=True, subset=['yesterday','sentiment_score'])

    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','yesterday']]
    X['weekofyear'] = X['weekofyear'].astype('int64')

    y = df["sentiment_score"]
    return X, y

def get_from_database():
    f = open("config.json", "r")
    values = json.load(f)
    f.close()
    database = values['DATABASE']
    user = values['USER']
    password = values['PWD']
    host = values['HOST']
    port = values['PORT']
    table = values['TABLE']
    engine = psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    ) 
    sql = "select * from "+table+";"
    return pd.io.sql.read_sql_query(sql, engine,parse_dates=['publish_datetime'])

def load_data():
    if os.path.isfile("9epoch.csv"):
        df = pd.read_csv("9epoch.csv",parse_dates=['publish_datetime'])
    else:
        df = get_from_database()
        df.to_csv("9epoch.csv")
    return df

def try_many_models():
    df = load_data()
    df = df.resample('d', on='publish_datetime')["sentiment_score"].mean().to_frame()
    X, y = create_features(df, label='sentiment_score')
    eta_values = [0.2,0.3,0.4]
    max_depth_values = [1,2,3,4,5,6,7,8,9,10]
    n_estimators_values = [50, 100, 150, 200, 250, 300, 500, 1000]
    reg_alpha_values = []
    reg_lambda_values = [0.8,1,1.2]
    tree_method_values = ['exact', 'approx', 'hist']

    if os.path.isfile("scores.txt"):
        os.remove("scores.txt")

    f = open("scores.txt",'a')

    i = 0
    max_score = -1.0
    for e in eta_values:
        for depth in max_depth_values:
            for ne in n_estimators_values:
                for l in reg_lambda_values:
                    for tree in tree_method_values:
                        # The random_state of the kfold should be different
                        # each time so we're not constantly training on the
                        # same data, which risks overfitting.
                        kfold = KFold(n_splits=5, shuffle=True, random_state=i)
                        i += 1
                        model = xgboost.XGBRegressor(base_score=0.5, booster='gbtree',
                           eta=e, max_depth=depth, n_estimators=ne,
                           objective='reg:squarederror', random_state=0,
                           reg_alpha=0, reg_lambda=l, verbosity=0,eval_metric='mae',
                           tree_method_values=tree)
                        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kfold)
                        error = sum(scores)/len(scores)
                        d = {"error":error,"eta":e,"max_depth":depth,"n_estimators":ne,"lambda":l,"tree":tree}
                        f.write(json.dumps(d))
                        f.write('\n')
                        max_score = max(max_score,error)
    f.close()
    print('\n\n Best score:',max_score,'\n\n')

def final_model(name,e,max_d,n_ests,a,l,tree):
    df = load_data()
    df = df.resample('d', on='publish_datetime')["sentiment_score"].mean().to_frame()
    X, y = create_features(df, label='sentiment_score')
    kfold = KFold(n_splits=5, shuffle=True, random_state=8)
    model = xgboost.XGBRegressor(base_score=0.5, booster='gbtree',
                    eta=e, max_depth=max_d, n_estimators=n_ests,
                    objective='reg:squarederror', random_state=0,
                    reg_alpha=a, reg_lambda=l, verbosity=0,eval_metric='mae',
                    tree_method_values=tree)
    model.fit(X,y)
    model.save_model(name)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kfold, n_jobs=-1)
    print("\n\n Mean score:",sum(scores)/len(scores))