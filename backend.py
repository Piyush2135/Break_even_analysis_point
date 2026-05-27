import numpy as np
import pandas as pd


def generate_data():
    np.random.seed(42)
    days = np.arange(1, 51)
    demand = 200 + 2*days + np.random.normal(0, 20, len(days))

    return pd.DataFrame({
        "Day": days,
        "Demand": demand
    })


def linear_regression(X, y):
    mean_x = np.mean(X)
    mean_y = np.mean(y)

    m = np.sum((X-mean_x)*(y-mean_y)) / np.sum((X-mean_x)**2)
    c = mean_y - m*mean_x

    return m, c


def predict_demand(m, c):
    future_days = np.arange(len(X)+1,len(X)+11)
    predicted = m*future_days + c
    return future_days, predicted


def break_even(FC, SP, VC):
    return FC / (SP - VC)


def calculate_profit(SP, VC, FC, demand):
    return (SP - VC)*demand - FC


def optimal_price(FC, VC, target_profit, demand):
    return VC + (FC + target_profit) / demand
