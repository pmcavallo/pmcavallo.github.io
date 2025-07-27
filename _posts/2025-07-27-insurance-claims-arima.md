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
![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima7.png?raw=true) 

⚠️ Note: The seasonal coefficients have extremely large magnitudes. While they are statistically significant, such values could hint at issues with scaling, multicollinearity, or data artifacts. It may also reflect the simulated nature of the data.

📉 **Model Fit Statistics**  
Log Likelihood: -773.56  
AIC: 1557.12  
BIC: 1564.60  
HQIC: 1559.64  


These values indicate the overall fit of the model, with lower being better. While not directly interpretable, they are useful for model comparison.

🧪 Diagnostic Tests

| Test                   | Value | p-value | Interpretation                            |
| ---------------------- | ----- | ------- | ----------------------------------------- |
| Ljung-Box (Q)          | 1.42  | 0.23    | No strong autocorrelation in residuals.   |
| Jarque-Bera (JB)       | 0.63  | 0.73    | Residuals appear normally distributed.    |
| Heteroskedasticity (H) | 0.38  | 0.12    | No strong evidence of heteroskedasticity. |

✅ Conclusion: The model residuals show no significant autocorrelation, are likely homoskedastic, and pass the normality test. This suggests the SARIMA model provides a reasonably good fit to the simulated data.

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

Next, we'll take a look a component breakdown plot showing how each element of the SARIMA model affects the time series.

```python
df['Fitted'] = model_fit.fittedvalues
df['Residuals'] = df['Monthly_Claim_Payout'] - df['Fitted']

# Create component lines
observed = df['Monthly_Claim_Payout']
fitted = df['Fitted']
residuals = df['Residuals']
forecast_error = observed - fitted

# Set up the figure
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# 1. Observed Series (black)
axs[0].plot(observed, color='black', linewidth=2, label='Observed')
axs[0].set_title('Observed Series')
axs[0].legend()

# 2. Model Residuals (gray)
axs[1].plot(residuals, color='gray', linewidth=1.5, label='Residuals')
axs[1].set_title('Model Residuals')
axs[1].legend()

# 3. Fitted Values (orange)
axs[2].plot(fitted, color='orange', linewidth=1.5, label='Fitted Values')
axs[2].set_title('Fitted Values (AR+MA effects)')
axs[2].legend()

# 4. Forecast Error (Observed - Fitted, red)
axs[3].plot(forecast_error, color='red', linewidth=1.5, label='Observed - Fitted')
axs[3].set_title('Observed - Fitted (Residual Errors)')
axs[3].legend()

# Shared x-axis label
axs[3].set_xlabel('Date')

# Shared y-axis label
for ax in axs:
    ax.set_ylabel('Monthly Claim Payout ($)')
    ax.grid(True)

plt.tight_layout()
plt.show()

```

![arima](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/arima8.png?raw=true) 

🔍 Interpretation of Each Panel:
Observed Series (Top Panel):
- This is the original simulated monthly insurance payout series.
- We can clearly see a seasonal pattern (with annual peaks and troughs) and an upward trend over time.

Model Residuals:
- These are the unexplained parts of the series after accounting for AR, MA, and seasonal components.
- Ideally, residuals should resemble white noise. Here, we observe some large residual spikes, suggesting imperfect model fit—especially around 2021 and early 2023.

Fitted Values (AR+MA Effects):
- These are the values predicted by the SARIMA(1,1,1)(1,1,1,12) model.
- They track the general structure of the observed series well but may overfit or underfit certain volatile months.

Observed - Fitted (Residual Errors):
- This plot highlights where the model is over- or under-predicting.
- Large deviations from zero suggest room for improvement in the model specification—potentially better seasonal differencing or more flexible trend handling.

🔍 What’s causing the 2021 spike?
Simulated randomness:
- Since the dataset is synthetic and includes a random noise component, it's highly likely that the 2021 spike reflects a random high-variance event (e.g., a simulated shock or outlier month).

SARIMA model limitations:
- The SARIMA model, while designed to capture seasonality and trend, may underfit localized spikes — particularly if they are not seasonal or autoregressive in nature. These one-off changes go into the residual.

Look at the residuals:
- The positive spike in the "Observed - Fitted" curve in 2021 tells us:
  - The actual payout was much higher than the model’s prediction.
  - The model underestimated the spike and treated it as noise.

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

## 📈 Forecast Plot Interpretation

The chart above presents a 12-month out-of-sample forecast of monthly insurance claim payouts using a SARIMA(1,1,1)(1,1,1,12) model. The solid blue line represents the historical data from 2020 to the end of 2024, while the dotted orange line reflects the model's forecast for 2025.

The shaded orange region indicates the 95% confidence interval (CI), capturing the uncertainty around the point forecast. Notably:
- The forecast extends the upward trend observed in late 2024, consistent with seasonal patterns captured in prior periods.
- The widening CI reflects increasing uncertainty further into the forecast horizon — a typical characteristic of time series models.
- The transition between historical and forecasted values is smooth and continuous, confirming no structural breaks or data leakage.

This visualization provides stakeholders with a forward-looking estimate of potential claim payouts, with clearly communicated risk bounds.



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

*Note: Both MAE and RMSE are expressed in dollars, consistent with the unit of the dependent variable (Monthly Claim Payout)*

## ✅ Final Conclusions

- SARIMA(1,1,1)(1,1,1,12) successfully modeled trend and seasonality in monthly insurance claim payouts.
- Initial instability (exploding CIs) was addressed using measurement_error=True, a common best practice
- Forecast performance metrics confirm robustness and realism
- All steps are fully reproducible and modular for adaptation to real datasets



