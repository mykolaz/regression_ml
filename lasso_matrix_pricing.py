from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV, RidgeCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

FS = (14, 6) # Figure Size
RS = 124 # Random State
N_JOBS = 8 # Number of parallel threads

# Repeated K-folds
N_SPLITS = 10
N_REPEATS = 1

# Optuna
N_TRIALS = 100
MULTIVARIATE = True

# XGBoost
EARLY_STOPPING_ROUNDS = 100

def remove_outliers_iqr(df, k=1.5): # Changing it to 2 from 1.5 | from 2.21 to 4
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


#dataset = pd.get_dummies(raw_dataset, columns=['ticker'], prefix='ticker_', prefix_sep='')
#df_cleaned = remove_outliers_z_score(raw_dataset)
df_cleaned = remove_outliers_iqr(raw_dataset)
#df_cleaned = raw_dataset

dataset_copy = df_cleaned.copy()

y = dataset_copy['ticker_todays_increase']
y = y*100
#y = round(y, 4)

#y = lab.fit_transform(y)
X = df_cleaned.drop('ticker_todays_increase', axis = 1)
#X = X.clip(lower=-1e20, upper=1e20)
#X.fillna(X.mean(), inplace=True)

lasso = ElasticNetCV(
        cv=10, 
        verbose=2,  # Set to 0 to avoid too much output, or keep 2 for detailed logging
        random_state=42,
        l1_ratio=0.75,
        alphas=np.logspace(-5, 1, 100),
        max_iter=110000,
        tol=0.001
    ).fit(X, y)

coef = pd.Series(lasso.coef_, index=X.columns)


# Set the threshold of the coefficient
max_coef = 0.03

selected_features = coef[coef > 0.0005].index

print(f"Selected features by Lasso: {selected_features.tolist()}")

ridge = RidgeCV(alphas=np.logspace(-5, 1, 100), cv=10).fit(X, y)
print("Ridge Coefficients:", ridge.coef_)

# Visualize the coefficients
# Here we plot the actual coefficient values for features that exceed max_coef
max_coef = 0.0005  # or whatever threshold you're using
coef_to_plot = coef[abs(coef) > max_coef]
coef_to_plot.sort_values().plot(kind="barh", title="Significant Lasso Coefficients")
plt.show()

# For selected features, we'll plot their corresponding coefficients
selected_coef = coef[selected_features]
selected_coef.sort_values().plot(kind="barh", title="Selected Lasso Features Coefficients")
plt.show()

# If you want to plot Ridge coefficients, you'll need to match them with feature names
# Assuming X.columns gives you the feature names in the correct order:
ridge_coef = pd.Series(ridge.coef_, index=X.columns)
ridge_coef.sort_values().plot(kind="barh", title="Ridge Coefficients")
plt.show()

print("Best Alpha:", lasso.alpha_)
print("Best L1 Ratio:", lasso.l1_ratio_)
print("MSE:", lasso.mse_path_.mean(axis=1).min())