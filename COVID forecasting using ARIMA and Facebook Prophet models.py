import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from pylab import rcParams

# Configuration and style settings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 18, 8
matplotlib.rcParams.update({'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'text.color': 'k'})

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def visualize_data(y):
    y.plot(figsize=(15, 6))
    plt.show()

    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    decomposition.plot()
    plt.show()

def select_arima_parameters(y):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

def fit_arima_model(y):
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(16, 10))
    plt.show()
    return results

def validate_forecasts(y, results):
    pred = results.get_prediction(start=pd.to_datetime('2021-08-01'), dynamic=False)
    pred_ci = pred.conf_int()

    ax = y['2020':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('ICU Active')
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_truth = y['2021-08-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print(f'The Mean Squared Error of our forecasts is {round(mse, 2)}')
    print(f'The Root Mean Squared Error of our forecasts is {round(np.sqrt(mse), 2)}')

def forecast_with_arima(y, results):
    pred_uc = results.get_forecast(steps=14)
    pred_ci = pred_uc.conf_int()

    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('ICU Active')
    plt.legend()
    plt.show()

def forecast_with_prophet(df):
    icu = df.reset_index()[['date', 'bc_icu_active']].rename(columns={'date': 'ds', 'bc_icu_active': 'y'})
    icu_model = Prophet(interval_width=0.95)
    icu_model.fit(icu)
    icu_forecast = icu_model.make_future_dataframe(periods=12, freq='MS')
    icu_forecast = icu_model.predict(icu_forecast)

    icu_model.plot(icu_forecast, xlabel='Date', ylabel='ICU')
    plt.title('COVID ICU')
    plt.show()
    icu_model.plot_components(icu_forecast)
    plt.show()

    df_cv = cross_validation(icu_model, initial='100 days', period='30 days',
