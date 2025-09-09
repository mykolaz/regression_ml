from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Equitable colour
HEX = '#00A4AC'

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

def remove_outliers_iqr(df, k=1.5): # Changing it to 2.21 from 1.5 | and from 2.21 to 4
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
                                'ticker_dow_annual_beta': np.float64,
                                'ticker_previous_close': np.float64
                          }
                        )


#df_cleaned = remove_outliers_z_score(raw_dataset)
df_cleaned = remove_outliers_iqr(raw_dataset)
#df_cleaned = raw_dataset

print (len(raw_dataset),len(df_cleaned))

#dataset = pd.get_dummies(raw_dataset, columns=['ticker'], prefix='ticker_', prefix_sep='')

dataset_copy = df_cleaned.copy()

y = dataset_copy['ticker_todays_increase']
y = y*100
#y = np.log(y+1)

X = df_cleaned.drop('ticker_todays_increase', axis = 1)

# Define the model
SAMPLES_BYLEVEL = 0.5638279715944908
SAMPLES_BY_NODE = 0.6788126131994223
SAMPLES_BY_TREE = 0.9999
GAMMA = 0.24303458184980659
LEARNING_RATE = 0.16584566868703912
MIN_CHILD_WEIGHT = 6.018730660518377
ALPHA = 0.2241247331074888
LAMBDA = 15.893248200595973
SUBSAMPLE = 0.4952076368141825

#Static
N_ESTIMATORS = 150 #9000
MAX_DEPTH = 6


# Train the XGBoost model
model = XGBRegressor(
    enable_categorical = True,
    objective = 'reg:squarederror', # reg:  absoluteerror | squarederror | pseudohubererror
    booster = 'gbtree',
    subsample = SUBSAMPLE,
    colsample_bynode = SAMPLES_BY_NODE,
    colsample_bytree = SAMPLES_BY_TREE,
    colsample_bylevel = SAMPLES_BYLEVEL,
    learning_rate = LEARNING_RATE,
    max_depth = MAX_DEPTH,
    alpha = ALPHA,
    min_child_weight = MIN_CHILD_WEIGHT,
    n_estimators = N_ESTIMATORS,
    reg_lambda = LAMBDA,
    gamma = GAMMA
    #min_samples_leaf = MIN_SAMPLES_LEAF
)

cv = KFold(n_splits=8, shuffle=True, random_state=42)

# Perform RFE with cross-validation
rfe = RFECV(model, step=1, cv=cv, scoring='neg_mean_squared_error', n_jobs=30, verbose=2) #'neg_mean_squared_error' | huberscorer
rfe.fit(X, y)

# Visualize the optimal number of features
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(rfe.cv_results_['mean_test_score']) + 1), rfe.cv_results_['mean_test_score'], color=HEX)
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (MSE)")
plt.title("Optimal Number of Features")
plt.show()

# Get the selected features
selected_features = list(X.columns[rfe.support_])
print(f"Selected features: {selected_features}")

print('\n\n',rfe.cv_results_['mean_test_score'])