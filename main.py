import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join('data','walmart_sales.csv')

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        print(f'WARNING: dataset not found at {path}. Please put walmart_sales.csv in the data/ folder.')
        return None
    return pd.read_csv(path, parse_dates=['date'])

def feature_engineer(df):
    df = df.copy()
    # Expecting columns: date, store, dept, weekly_sales
    df = df.sort_values('date')
    df['weekofyear'] = df['date'].dt.isocalendar().week
    # Aggregate to weekly total sales per store (example)
    agg = df.groupby(['date']).agg({'weekly_sales':'sum'}).reset_index()
    agg['lag_1'] = agg['weekly_sales'].shift(1).fillna(method='bfill')
    agg['lag_2'] = agg['weekly_sales'].shift(2).fillna(method='bfill')
    X = agg[['lag_1','lag_2']]
    y = agg['weekly_sales']
    return X,y

def train_and_eval(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    model = LinearRegression()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test,preds))
    print('RMSE:', mean_squared_error(y_test,preds, squared=False))
    # simple plot
    plt.plot(y_test.values, label='actual')
    plt.plot(preds, label='predicted')
    plt.legend()
    plt.title('Walmart sales - actual vs predicted')
    plt.show()

def main():
    df = load_data()
    if df is None:
        return
    X,y = feature_engineer(df)
    train_and_eval(X,y)

if __name__ == '__main__':
    main()
