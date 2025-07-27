# Forecasting Monthly Insurance Claim Payouts Using ARIMA  
**Project Date**: July 27, 2025  
**Author**: Simulated Insurance Analytics Project  
**Modeling Objective**: Forecast monthly dollar-denominated insurance claim payouts for the year 2025 using ARIMA modeling, with full diagnostics and evaluation.

---

## 📌 Project Overview

This project simulates realistic monthly insurance claim payouts over a 5-year period (2020–2024), applies ARIMA modeling for time series forecasting, and evaluates model performance on a simulated 2025 out-of-sample dataset.

---

## 📊 Step 1: Simulate Monthly Insurance Claim Data

We simulate claim payouts using:
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

---

(remaining content truncated for brevity here...)

