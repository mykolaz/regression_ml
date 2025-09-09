
from scipy.stats import uniform

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier, DMatrix
from optuna import create_study
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import xgboost as xgb
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization
from numpy import sqrt
from scipy import stats

def huber_loss_scorer(y_true, y_pred, delta=0.5):
    residual = np.abs(y_true - y_pred)
    is_small_error = residual <= delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * residual - 0.5 * delta ** 2
    loss = np.where(is_small_error, squared_loss, linear_loss)
    return np.mean(loss)  # Negative for higher-is-better

huber_scorer = make_scorer(huber_loss_scorer, greater_is_better=False, delta=0.5)

def remove_outliers_z_score(df, threshold=3):
    # Compute Z-score for each column
    z_scores = stats.zscore(df)
    
    # Create a boolean mask where True means the value is not an outlier
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    
    # Return the dataframe without outliers
    return df[filtered_entries]

def remove_outliers_iqr(df, k=1.5): # Changing to 2.21 from 1.5 | 1.618
    outliers = pd.DataFrame()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        outliers[col] = ~((df[col] >= lower_bound) & (df[col] <= upper_bound))
    
    # Remove rows where any column has an outlier
    return df[~outliers.any(axis=1)]

# Prepare training and testing datasets
raw_dataset = pd.read_csv('market_data.csv',
                          dtype= {
                                #'ticker': str,
                                'ticker_prior_days_increase': np.float64,                                
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
                                'ticker_dow_annual_beta': np.float64
                          }
                        )


#dataset = pd.get_dummies(raw_dataset, columns=['ticker'], prefix='ticker_', prefix_sep='')


#df_cleaned = remove_outliers_z_score(raw_dataset)
df_cleaned = remove_outliers_iqr(raw_dataset)
#df_cleaned = raw_dataset

print (len(raw_dataset),len(df_cleaned))

#dataset = pd.get_dummies(raw_dataset, columns=['ticker'], prefix='ticker_', prefix_sep='')

dataset_copy = df_cleaned.copy()

y = dataset_copy['ticker_todays_increase']
y = y*100
#y = round(y, 4)

#y = lab.fit_transform(y)
X = df_cleaned.drop('ticker_todays_increase', axis = 1)
#X = X.clip(lower=-1e20, upper=1e20)
#X.fillna(X.mean(), inplace=True)

# with max_depth of 7
#X = X[['four_month_support_trend_value', 'four_month_resistance_trend_value', 'days_since_dma_flip', 'six_month_resistance', 'six_month_support', 'six_month_support_trend_value', 'six_month_resistance_trend_value', 'three_month_resistance', 'three_month_support', 'three_month_support_trend_value', 'three_month_resistance_trend_value', 'twelve_month_resistance', 'twelve_month_support', 'twelve_month_support_trend_value', 'twelve_month_resistance_trend_value', 'bollinger_high', 'bollinger_low', 'weekly_bollinger_high', 'weekly_bollinger_low', 'dma_15', 'one_week_resistance', 'one_week_support', 'one_week_support_trend_value', 'one_week_resistance_trend_value', 'two_week_resistance', 'two_week_support', 'two_week_support_trend_value', 'two_week_resistance_trend_value', 'one_month_resistance', 'one_month_support', 'one_month_support_trend_value', 'one_month_resistance_trend_value', 'two_month_resistance', 'two_month_support', 'two_month_resistance_trend_value', 'two_month_support_trend_value', 'dma_200_50_difference', 'dma_200_50_difference_slope', 'dma_50', 'dma_200', 'dma_50_slope', 'dma_200_slope', 'days_since_dma_flip_2', 'ticker_audit_risk', 'ticker_board_risk', 'ticker_compensation_risk', 'ticker_shareholder_rights_risk', 'ticker_overall_risk', 'ticker_price_hint', 'ticker_dividend_yield', 'ticker_beta', 'ticker_trailing_pe', 'ticker_forward_pe', 'ticker_52w_low', 'ticker_52w_high', 'ticker_profit_margins', 'ticker_short_ratio', 'ticker_short_percent_of_float', 'ticker_price_to_book', 'ticker_earnings_quarterly_growth', 'ticker_trailing_eps', 'ticker_forward_eps', 'ticker_peg_ratio', 'ticker_enterprise_to_revenue', 'ticker_enterprise_to_ebitda', 'tiker_52w_change', 'ticker_target_high_price', 'ticker_target_low_price', 'ticker_target_mean_price', 'ticker_target_median_price', 'ticker_total_cash_per_share', 'ticker_quick_ratio', 'ticker_current_ratio', 'ticker_revenue_per_share_ratio', 'ticker_earnings_growth', 'ticker_revenue_growth', 'ticker_gross_margins', 'ticker_ebidta_margins', 'ticker_operating_margins', 'ticker_trailing_peg_ratio', 'ticker_recommendation_average', 'fibonacci_0', 'fibonacci_38_2', 'fibonacci_50', 'fibonacci_61_8', 'fibonacci_100', 'daily_rsi', 'daily_rsi_slope', 'daily_macd', 'daily_macd_line', 'daily_macd_slope', 'daily_macd_difference', 'weekly_rsi', 'weekly_rsi_slope', 'weekly_macd', 'weekly_macd_line', 'weekly_macd_slope', 'weekly_macd_difference', 'daily_macd_rsi_slope_difference', 'weekly_macd_rsi_slope_difference', 'current_lower_band', 'current_upper_band', 'days_since_supertrend', 'previous_lower_band', 'previous_upper_band', 'supertrend_value', 'fear_and_greed_value', '_10d_bullish', '_9d_bullish', '_8d_bullish', '_7d_bullish', '_6d_bullish', '_5d_bullish', '_4d_bullish', '_3d_bullish', '_2d_bullish', '_1d_bullish', '_3d_bullish_trend', '_5d_bullish_trend', '_10d_bullish_trend', 'max_pain_1', 'from_days_1', 'pc_1', 'max_pain_2', 'from_days_2', 'pc_2', 'max_pain_3', 'from_days_3', 'pc_3', 'max_pain_4', 'from_days_4', 'pc_4', 'max_pain_5', 'from_days_5', 'pc_5', 'di_plus', 'di_minus', 'adx', 'pk_15', 'pk_14', 'pk_13', 'pk_12', 'pk_11', 'pk_10', 'pk_9', 'pk_8', 'pk_7', 'pk_6', 'pk_5', 'pk_4', 'pk_3', 'pk_2', 'pk_1', 'pd_15', 'pd_14', 'pd_13', 'pd_12', 'pd_11', 'pd_10', 'pd_9', 'pd_8', 'pd_7', 'pd_6', 'pd_5', 'pd_4', 'pd_3', 'pd_2', 'pd_1', '_3d_slope_15', '_3d_slope_14', '_3d_slope_13', '_3d_slope_12', '_3d_slope_11', '_3d_slope_10', '_3d_slope_9', '_3d_slope_8', '_14d_slope_7', '_14d_slope_6', '_14d_slope_5', '_14d_slope_4', '_14d_slope_3', '_14d_slope_2', '_14d_slope_1', 'psar_af', 'psar_ep', 'psar', 'psar_trend', 'previous_psar_af', 'previous_psar_ep', 'previous_psar', 'previous_psar_trend', 'benchmark_alpha', 'benchmark_beta', 'benchmark_beta_offset', 'benchmark_other_beta', 'benchmark_other_offset', 'benchmark_p', 'benchmark_r2', 'previous_vwap', 'previous_weekly_vwap', 'vwap', 'weekly_vwap', 'max_call_vega_1', 'max_call_vega_2', 'max_call_vega_3', 'max_call_vega_4', 'max_call_vega_5', 'max_put_vega_1', 'max_put_vega_2', 'max_put_vega_3', 'max_put_vega_4', 'max_put_vega_5', 'max_call_gamma_1', 'max_call_gamma_2', 'max_call_gamma_3', 'max_call_gamma_4', 'max_call_gamma_5', 'max_put_gamma_1', 'max_put_gamma_2', 'max_put_gamma_3', 'max_put_gamma_4', 'max_put_gamma_5', 'max_call_oi_1', 'max_call_oi_2', 'max_call_oi_3', 'max_call_oi_4', 'max_call_oi_5', 'max_put_oi_1', 'max_put_oi_2', 'max_put_oi_3', 'max_put_oi_4', 'max_put_oi_5', 'max_call_oi_pct_1', 'max_call_oi_pct_2', 'max_call_oi_pct_3', 'max_call_oi_pct_4', 'max_call_oi_pct_5', 'max_put_oi_pct_1', 'max_put_oi_pct_2', 'max_put_oi_pct_3', 'max_put_oi_pct_4', 'max_put_oi_pct_5', 'itm_call_delta_1', 'itm_call_delta_2', 'itm_call_delta_3', 'itm_call_delta_4', 'itm_call_delta_5', 'itm_call_gamma_1', 'itm_call_gamma_2', 'itm_call_gamma_3', 'itm_call_gamma_4', 'itm_call_gamma_5', 'itm_call_theta_1', 'itm_call_theta_2', 'itm_call_theta_3', 'itm_call_theta_4', 'itm_call_theta_5', 'itm_call_vega_1', 'itm_call_vega_2', 'itm_call_vega_3', 'itm_call_vega_4', 'itm_call_vega_5', 'itm_call_rho_1', 'itm_call_rho_2', 'itm_call_rho_3', 'itm_call_rho_4', 'itm_call_rho_5', 'itm_put_delta_1', 'itm_put_delta_2', 'itm_put_delta_3', 'itm_put_delta_4', 'itm_put_delta_5', 'itm_put_gamma_1', 'itm_put_gamma_2', 'itm_put_gamma_3', 'itm_put_gamma_4', 'itm_put_gamma_5', 'itm_put_theta_1', 'itm_put_theta_2', 'itm_put_theta_3', 'itm_put_theta_4', 'itm_put_theta_5', 'itm_put_vega_1', 'itm_put_vega_2', 'itm_put_vega_3', 'itm_put_vega_4', 'itm_put_vega_5', 'itm_put_rho_1', 'itm_put_rho_2', 'itm_put_rho_3', 'itm_put_rho_4', 'itm_put_rho_5', 'max_pain_call_delta_1', 'max_pain_call_delta_2', 'max_pain_call_delta_3', 'max_pain_call_delta_4', 'max_pain_call_delta_5', 'max_pain_call_gamma_1', 'max_pain_call_gamma_2', 'max_pain_call_gamma_3', 'max_pain_call_gamma_4', 'max_pain_call_gamma_5', 'max_pain_call_theta_1', 'max_pain_call_theta_2', 'max_pain_call_theta_3', 'max_pain_call_theta_4', 'max_pain_call_theta_5', 'max_pain_call_vega_1', 'max_pain_call_vega_2', 'max_pain_call_vega_3', 'max_pain_call_vega_4', 'max_pain_call_vega_5', 'max_pain_call_rho_1', 'max_pain_call_rho_2', 'max_pain_call_rho_3', 'max_pain_call_rho_4', 'max_pain_call_rho_5', 'max_pain_put_delta_1', 'max_pain_put_delta_2', 'max_pain_put_delta_3', 'max_pain_put_delta_4', 'max_pain_put_delta_5', 'max_pain_put_gamma_1', 'max_pain_put_gamma_2', 'max_pain_put_gamma_3', 'max_pain_put_gamma_4', 'max_pain_put_gamma_5', 'max_pain_put_theta_1', 'max_pain_put_theta_2', 'max_pain_put_theta_3', 'max_pain_put_theta_4', 'max_pain_put_theta_5', 'max_pain_put_vega_1', 'max_pain_put_vega_2', 'max_pain_put_vega_3', 'max_pain_put_vega_4', 'max_pain_put_vega_5', 'max_pain_put_rho_1', 'max_pain_put_rho_2', 'max_pain_put_rho_3', 'max_pain_put_rho_4', 'max_pain_put_rho_5', 'last_known_dma_flip_Negative', 'last_known_dma_flip_Positive', 'last_known_dma_flip_2_Negative', 'last_known_dma_flip_2_Positive', 'recommendation_key_buy', 'recommendation_key_hold', 'recommendation_key_none', 'recommendation_key_strong_buy', 'supertrend_direction_BUY', 'supertrend_direction_SELL', 'fear_and_greed_description_fear']]

X = X[['ticker_prior_days_increase', 'ticker_weekly_return', 'sp500_daily_return', 'sp500_weekly_return', 'sp500_monthly_return', 'sp500_ytd_return', 'sp500_annual_return', 'ndaq_daily_return', 'ndaq_weekly_return', 'ndaq_ytd_return', 'dow30_daily_return', 'dow30_weekly_return', 'dow30_monthly_return', 'dow30_annual_return', 'ticker_sp500_weekly_beta', 'ticker_ndaq_weekly_beta', 'ticker_dow_weekly_beta']]

y_min = min(y)
y_max = max(y)

print ('Min: ', y_min)
print ('Max: ', y_max)

#non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function for Bayesian Optimization
def xgb_cv(subsample, learning_rate, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, gamma, reg_lambda, min_child_weight):

    #n_estimators = int(n_estimators)
    #max_depth = int(max_depth)
    n_estimators = 150 #1089 #int(n_estimators) #1000
    max_depth = 6 #int(max_depth) #5

    params = {
        'n_estimators': n_estimators,  # or use a larger number and use early stopping
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'gamma': gamma,
        'objective': 'reg:squarederror',  # Use  reg:  absoluteerror | squarederror | pseudohubererror
        'random_state': 42,
        'booster': 'gbtree',
        'min_child_weight': min_child_weight,
        'grow_policy': 'lossguide'
    }
    
    
    # Perform cross-validation
    model = xgb.XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=8, n_jobs=10) #'neg_mean_squared_error' | huber_scorer

    #mape_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_percentage_error', cv=10, n_jobs=3)

    return cv_scores.mean()#, -np.mean(mape_scores)

# Set the parameter bounds for Bayesian Optimization
param_bounds = {
    'subsample': (0.2, 0.9999),
    'colsample_bytree': (0.2, 0.9999),
    'colsample_bylevel': (0.2, 0.9999),
    'colsample_bynode': (0.2, 0.9999),
    'reg_alpha': (0.001, 0.9999),
    'gamma': (0.001, 50.00),
    'reg_lambda': (0.001, 50.0),
    'min_child_weight': (0.01, 70),
    'learning_rate': (0.00001, 0.9),
    #'n_estimators': (400, 2000),
    #'max_depth': (1, 20)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=3, n_iter=47)

# Best parameters found
print("Best parameters found:")
print(optimizer.max)

print (sqrt(-1*optimizer.max['target']))