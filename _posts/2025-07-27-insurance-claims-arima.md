---
layout: post
title: Forecasting Monthly Insurance Claim Payouts Using ARIMA
date: 2024-03-12
---

This project simulates realistic monthly insurance claim payouts over a 5-year period (2020–2024), applies ARIMA modeling for time series forecasting, and evaluates model performance on a simulated 2025 out-of-sample dataset.

---

## 📊 Step 1: Simulate Monthly Insurance Claim Data

I simulate claim payouts using:
- An upward linear trend  
- Annual seasonality (e.g., winter storms, hurricane season)  
- Gaussian noise

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Time index
dates = pd.date_range(start="2020-01-01", end="2024-12-01", freq="MS")
n_periods = len(dates)

# Components
trend = np.linspace(100000, 130000, n_periods)
seasonality = 10000 * np.sin(2 * np.pi * dates.month / 12)
noise = np.random.normal(0, 5000, n_periods)

# Simulated payouts
monthly_claims = trend + seasonality + noise
monthly_claims = np.round(monthly_claims, 2)

df = pd.DataFrame({ 'Date': dates, 'Monthly_Claim_Payout': monthly_claims })
df.set_index('Date', inplace=True)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Monthly_Claim_Payout'], label="Monthly Claims ($)")
plt.title("Simulated Monthly Insurance Claim Payouts")
plt.ylabel("USD")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima.png?raw=true) 

🔍 Interpretation of the Simulated Series

- Upward Trend: There's a modest increase over time, capturing effects like inflation, population growth, or policy expansion.
- Clear Seasonality: Peaks and troughs recur annually — possibly reflecting seasonal risks (e.g., winter storm damage in Q1, hurricane claims in Q3).
- Irregular Noise: Random fluctuations ensure that the series isn't overly smooth, mimicking month-to-month volatility.

---

## 📉 Step 2: Decomposition, ADF Test, ACF/PACF

I decompose the series and test for stationarity.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Decompose
decomposition = seasonal_decompose(df['Monthly_Claim_Payout'], model='additive', period=12)
decomposition.plot()
plt.suptitle('Seasonal Decomposition of Monthly Claim Payouts')
plt.tight_layout()
plt.show()

# ADF test
adf_result = adfuller(df['Monthly_Claim_Payout'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima2.png?raw=true) 

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima6.png?raw=true) 

📉 Step 2: Decomposition, ADF Test, ACF/PACF

The decomposition plot breaks the time series into three main components:

- Trend: Shows a clear upward slope, indicating a gradual increase in claim payouts over the years—possibly due to inflation, rising costs of services, or increased policyholder count.
- Seasonal: Displays a pronounced repeating pattern with a 12-month cycle, confirming the presence of annual seasonality—e.g., spikes in certain months might be linked to natural disaster seasons or end-of-year policy renewals.
- Residual: Appears randomly distributed around zero, suggesting that after removing trend and seasonality, no major patterns remain.

📌 Takeaway: The series exhibits non-stationary behavior with strong seasonality, making it a good candidate for seasonal differencing in SARIMA modeling.

I then run the Dickey-Fuller (ADF) test to formally assess stationarity. With a p-value of 0.9917, we fail to reject the null hypothesis that a unit root is present. This means the series is non-stationary — consistent with the upward trend we saw earlier.

The ACF and PACF plots guide the model order selection:

- ACF (Autocorrelation Function): Shows strong positive autocorrelations at lags 1 through 12, gradually declining — a signature of a seasonal trend and non-stationarity.
- PACF (Partial Autocorrelation Function): Displays a sharp cutoff after lag 1, suggesting a potential AR(1) process.

🔧 Modeling Implication: These patterns suggest an ARIMA model with differencing and seasonal components. 

## 🔁 Step 3: Differencing and ACF/PACF

I apply both regular and seasonal differencing, then check the ACF/PACF plots to inform model structure.

```python
# First and seasonal differencing
df_diff = df['Monthly_Claim_Payout'].diff().dropna()
df_diff_seasonal = df_diff.diff(12).dropna()

# ADF again
from statsmodels.tsa.stattools import adfuller
adf_diff = adfuller(df_diff_seasonal)
print(f"ADF Statistic (diff): {adf_diff[0]}")
print(f"p-value (diff): {adf_diff[1]}")

# Plot ACF/PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(df_diff_seasonal, lags=23, ax=ax[0])
plot_pacf(df_diff_seasonal, lags=23, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()
```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima3.png?raw=true) 

🔬 ADF Test (Differenced Series)

ADF Statistic: -1.3429
p-value: 0.6093
Critical Values:
   1%: -3.6267
   5%: -2.9460
  10%: -2.6117
  
Despite the differencing, the p-value remains high (0.6093) and the ADF statistic is above all critical values, indicating that the series is still non-stationary. This suggests that further differencing and/or seasonal differencing may be necessary.

📉 ACF and PACF (After Differencing)

- ACF: Rapid drop after lag 1, with most values inside the confidence interval — suggesting that the series is becoming less autocorrelated.
- PACF: Mostly insignificant after lag 1, with only lag 0 and perhaps lag 6/7 showing moderate significance.

🧠 Interpretation: First differencing reduced autocorrelation but wasn't sufficient to fully stationarize the series. I may need seasonal differencing (D=1, m=12) in our SARIMA model to fully capture the seasonality and achieve stationarity.

Summary:
After differencing, stationarity improves. ACF suggests MA(1), PACF suggests AR(1). Seasonal patterns persist. We proceed with SARIMA(1,1,1)(1,1,1,12).

---

## 🧠 Step 4: Fit SARIMA Model and Diagnose

We fit a SARIMA(1,1,1)(1,1,1,12) model and apply stabilization using `measurement_error=True` to avoid numerical issues such as exploding confidence intervals. This reflects best practices when working with synthetic or highly predictable series.

```python
import statsmodels.api as sm

model = sm.tsa.statespace.SARIMAX(
    df['Monthly_Claim_Payout'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=True,
    enforce_invertibility=True,
    measurement_error=True
)
results = model.fit()
print(results.summary())

```

📉 Custom Residual Diagnostics
I replace the default SARIMAX diagnostics with a custom 2x2 panel:

- Time series of residuals
- Histogram + KDE
- Normal Q-Q plot
- Residual autocorrelation (ACF)



```python
residuals = results.resid
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Residual time series
ax[0, 0].plot(residuals)
ax[0, 0].set_title("Standardized Residuals")
ax[0, 0].axhline(0, color='gray', linestyle='--')

# Histogram + KDE
sns.histplot(residuals, kde=True, ax=ax[0, 1], color='steelblue')
ax[0, 1].set_title("Residual Distribution with KDE")

# Q-Q plot
sm.qqplot(residuals, line='s', ax=ax[1, 0])
ax[1, 0].set_title("Normal Q-Q")

# Residual ACF
sm.graphics.tsa.plot_acf(residuals, lags=20, ax=ax[1, 1])
ax[1, 1].set_title("Residual ACF")

plt.suptitle("Custom Model Diagnostics: SARIMA(1,1,1)(1,1,1,12)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

```
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima4.png?raw=true) 

Interpretation:

- Residuals are white noise (no autocorrelation)
- Distribution is approximately normal
- No visible structure or trend in residuals
- ✅ This model is ready for forecasting


---

## 📈 Step 5: Forecast Next 12 Months

I now forecast the next 12 months (2025-01 to 2025-12) using our validated model and visualize the results with a confidence interval.

```python
# Forecast next 12 months
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
forecast_ci.columns = ['Lower_CI', 'Upper_CI']

# Create forecast DataFrame
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean.values,
    'Lower_CI': forecast_ci['Lower_CI'].values,
    'Upper_CI': forecast_ci['Upper_CI'].values
}, index=future_dates)

# Concatenate for smooth plotting
full_series = pd.concat([df['Monthly_Claim_Payout'], forecast_df['Forecast']])

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(full_series.index, full_series, label='Forecast', linestyle='--', color='darkorange')
plt.plot(df.index, df['Monthly_Claim_Payout'], label='Historical', color='steelblue')
plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                 color='orange', alpha=0.3, label='95% CI')
plt.title("12-Month Forecast of Insurance Claim Payouts (Stabilized SARIMA)")
plt.xlabel("Date")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima5.png?raw=true) 

Interpretation:

- Forecast line smoothly continues from historical series
- 95% confidence interval shows growing uncertainty, but remains realistic due to model stabilization


## 🧪 Step 6: Simulate True 2025 Values and Evaluate Forecast Accuracy

To validate model accuracy, I simulate plausible “true” values for 2025 using the same trend, seasonality, and noise structure as the training set. Then I compare the model’s forecast to these synthetic truths using MAE and RMSE.

```python

# Simulate true values for 2025
np.random.seed(123)
future_trend = np.linspace(135000, 145000, 12)
future_seasonality = 10000 * np.sin(2 * np.pi * future_dates.month / 12)
future_noise = np.random.normal(0, 5000, 12)
true_2025_values = np.round(future_trend + future_seasonality + future_noise, 2)

# Evaluation
mae = mean_absolute_error(true_2025_values, forecast_df['Forecast'].values)
rmse = np.sqrt(mean_squared_error(true_2025_values, forecast_df['Forecast'].values))

print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
```

Evaluation Results:

- Mean Absolute Error (MAE): $7,184.73
- Root Mean Squared Error (RMSE): $8,414.35
- ✅ These are strong performance metrics given the claim scale (~$120–145k/month), confirming the model generalizes well even on synthetic holdout data.



##✅ Final Conclusions

- SARIMA(1,1,1)(1,1,1,12) successfully modeled trend and seasonality in monthly insurance claim payouts.
- Initial instability (exploding CIs) was addressed using measurement_error=True, a common best practice
- Forecast performance metrics confirm robustness and realism
- All steps are fully reproducible and modular for adaptation to real datasets



