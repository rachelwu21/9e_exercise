import pandas as pd
import xgboost
import datetime
import psycopg2
import numpy as np
from flask import Flask, request, render_template
from flask.helpers import send_file
import os

def get_data(database,user,password,host,port,table):
    engine = psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    sql = "select * from "+table+";"
    df = pd.io.sql.read_sql_query(sql, engine, parse_dates=['publish_datetime'])
    return df

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    host = request.form['server']
    port = request.form['port']
    database = request.form['database']
    user = request.form['user']
    password = request.form['pwd']
    table = request.form['table']

    df = get_data(database,user,password,host,port,table)
    df = df.resample('d', on='publish_datetime')["sentiment_score"].mean().to_frame()
    
    model = xgboost.Booster({'nthread': None})
    model.load_model('home/ubuntu/app/final.model')
    date = pd.Series([1],index=[df.index.values[-1] + np.timedelta64(1, 'D')])
    next_x = pd.DataFrame.from_dict({
        'dayofweek': date.index.dayofweek,
        'quarter': date.index.quarter,
        'month': date.index.month,
        'year': date.index.year,
        'dayofyear': date.index.dayofyear,
        'dayofmonth': date.index.day,
        'weekofyear': date.index.isocalendar().week.astype('int64'),
        'yesterday':df["sentiment_score"].values[-1]
    }, orient='columns')[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','yesterday']]
    y = df["sentiment_score"]

    t = xgboost.DMatrix(next_x, label=None)
    y_pred = model.predict(t)
    y_pred = pd.Series(data=[y_pred[0]], index=[pd.to_datetime(date.index[0])])
    y.index = pd.to_datetime(y.index)
    y = y.append(to_append=y_pred, ignore_index=False)
    y.to_csv("/home/ubuntu/app/time_series.csv", index_label="date",header=["sentiment_score"])
    return send_file('/home/ubuntu/app/time_series.csv',
                mimetype='text/csv',
                attachment_filename='time_series.csv',
                as_attachment=True)
if __name__ == "__main__":
    app.run()
