import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Equitable colour
HEX = '#00A4AC'

# Prepare training and testing datasets
raw_dataset = pd.read_csv('market_data.csv',
                          dtype= {
                                'ticker_todays_increase': np.float64,
                                'ticker_weekly_return': np.float64,
                                'ticker_monthly_return': np.float64,
                                'ticker_ytd_return': np.float64,
                                'ticker_annual_return': np.float64,
                                'sp500_daily_return': np.float64,
                                'sp500_weekly_return': np.float64,
                                'sp500_monthly_return': np.float64,
                                'sp500_ytd_return': np.float64,
                                'sp500_annual_return': np.float64,
                                'ndaq_daily_return': np.float64,
                                'ndaq_weekly_return': np.float64,
                                'ndaq_monthly_return': np.float64,
                                'ndaq_ytd_return': np.float64,
                                'ndaq_annual_return': np.float64,
                                'dow30_daily_return': np.float64,
                                'dow30_weekly_return': np.float64,
                                'dow30_monthly_return': np.float64,
                                'dow30_ytd_return': np.float64,
                                'dow30_annual_return': np.float64,
                                'ticker_sp500_weekly_beta': np.float64,
                                'ticker_sp500_monthly_beta': np.float64,
                                'ticker_sp500_ytd_beta': np.float64,
                                'ticker_sp500_annual_beta': np.float64,
                                'ticker_ndaq_weekly_beta': np.float64,
                                'ticker_ndaq_monthly_beta': np.float64,
                                'ticker_ndaq_ytd_beta': np.float64,
                                'ticker_ndaq_annual_beta': np.float64,
                                'ticker_dow_weekly_beta': np.float64,
                                'ticker_dow_monthly_beta': np.float64,
                                'ticker_dow_ytd_beta': np.float64,
                                'ticker_dow_annual_beta': np.float64,
                                'ticker_previous_close': np.float64
                          }
                        )


#dataset = pd.get_dummies(raw_dataset, columns=['ticker'], prefix='ticker_', prefix_sep='')

dataset_copy = raw_dataset.copy()

y = dataset_copy['ticker_todays_increase']
y = y*50

X = raw_dataset.drop('ticker_todays_increase', axis = 1)

correlation_matrix = dataset_copy.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()