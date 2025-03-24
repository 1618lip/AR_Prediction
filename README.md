<h1 align='center'>
  AR-based Prediction with Differencing on Stock Prices
</h1>
<p align='center'>
  <a href="#"><img alt="" src="https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" /></a>
   <a href="#"><img alt="" src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" /></a>
   <a href="#"><img alt="" src="https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white" /></a>
   <a href="#"><img alt="" src="https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" /></a>
   <a href="#"><img alt="" src="https://img.shields.io/badge/LaTeX-47A141?style=for-the-badge&logo=LaTeX&logoColor=white" /></a>
</p>

## 1. Overview
<p align='center'>

This project demonstrates how to:

1. **Generate a synthetic stock price series** using Geometric Brownian Motion (GBM).  
2. **Transform the non-stationary data** by differencing, turning it into a (more) stationary series.  
3. **Fit an Autoregressive (AR) model** on the differenced series and perform multi-step forecasting.  
4. **Reconstruct (integrate) the forecasted differences** to get predictions in the original price scale.  
5. **Perform model order selection** (choose the best AR order based on a validation set).  
6. **Compute error metrics** (MSE, RMSE, MAPE) to evaluate forecasts.  
7. **Animate the best-model forecast** in Python.
</p>
<br />


---

## 2. Motivation

1. **Non-Stationary Data:**  
   Stock prices typically follow a non-stationary process (they can trend or drift over time). Standard AR models assume stationarity. To reconcile this, we either take **log-returns** or **first differences**. Here, we chose differencing to produce a stationary-like series.

2. **AR Model Simplicity:**  
   Autoregressive models are straightforward to implement (Levinson-Durbin for coefficients) and interpret. They can still capture short-range correlations in the data.

3. **Model Order Selection:**  
   The “best” AR order is not always obvious. We systematically try $\text{AR(m)}, \text{AR(m+1)},...,\text{AR(p)}$ and pick the one that yields the lowest forecast error (MSE) on a hold-out validation set. Here $m=20$, which is reasonable since low order model won't be good. 

4. **Error Metrics:**  
   - **MSE** (Mean Squared Error):  
     $\text{MSE} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2.$
   - **RMSE** (Root MSE):  
     $\text{RMSE} = \sqrt{\text{MSE}}.$
   - **MAPE** (Mean Absolute Percentage Error):  
     $\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|.$
   These help us see how close forecasts are to actual values.
---

## 3. Theoretical Background

### 3.1 Geometric Brownian Motion (GBM)

We generate synthetic stock prices $\{P_t\}$ via:

$$P_{t+1} = P_t \times e^{(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z_t},$$

where $\mu$ is the drift, $\sigma$ is the volatility, $\Delta t$ is the time increment (e.g., $1/252$ for daily in a 252-trading-day year), and $Z_t \sim \mathcal{N}(0,1)$.

### 3.2 Differencing to Achieve Stationarity

Raw stock prices are non-stationary. We apply **first differencing**:

$$d_t = P_{t+1} - P_t.$$

This differenced series $\{d_t\}$ often behaves more like a stationary process (assuming drift and slow changes). An AR model can then be fit to $\{d_t\}$.

### 3.3 Autoregressive Model (AR(\(p\)))

An $\text{AR}(p)$ model for the differenced series $\{d_t\}$ is:

  $$d_{t} = \phi_1 d_{t-1} + \phi_2 d_{t-2} + \cdots + \phi_p d_{t-p} + \epsilon_t,$$
  
where $\epsilon_t$ is white noise. We use **Levinson-Durbin** recursion to solve for $\phi_1, \ldots, \phi_p$.

### 3.4 Forecast Reconstruction (Integration)

Once we forecast $\hat{d}_{t+1}, \hat{d}_{t+2}, \ldots$ in differenced space, we reconstruct the actual price by cumulatively adding these differences to the last known price $P_{T}$:

$$\hat{P}_{T+1} = P_T + \hat{d}_{T+1},\quad\hat{P}_{T+2} = \hat{P}_{T+1} + \hat{d}_{T+2},\quad \dots$$
  
This “integration” step returns us to the original scale.

### 3.5 Model Order Selection

We try AR orders $1,2,\ldots,\text{maxOrder}$. For each order:
1. Fit $\text{AR}(p)$ on differenced data.
2. Forecast the next $k$ steps (the validation period).
3. Integrate to reconstruct $\hat{P}$.
4. Compute MSE, RMSE, MAPE vs. actual $P$.
5. Pick the order that yields the lowest MSE (or another chosen metric).

---

## 4. Implementation Details

### 4.1 C++ Code Structure

1. **`SyntheticDataGenerator.cpp`:**  
   Generates GBM prices.  
2. **`ARModel.cpp`:**  
   - Implements the Levinson-Durbin recursion to compute AR(\(p\)) coefficients.  
   - Provides functions for one-step and multi-step forward predictions in differenced space.  
3. **`main.cpp`:**  
   - Generates `fullPrices` via GBM.  
   - Splits into `trainPrices` (first 260 days) and `validPrices` (remaining days).  
   - **Differencing**: Creates `diffData` from `trainPrices`.  
   - **Model Selection**: For each AR order in $[1,\text{maxOrder}]$, fits the model, forecasts, integrates, and computes errors. Chooses the best AR order.  
   - **Outputs**:  
     - `forecasted_prices.txt`, `actual_future_prices.txt`, `train_prices.txt`, etc.  
     - Time indices for training (`train_time_indices.txt`) and forecast horizon (`forecast_time_indices.txt`).  
     - Error metrics vs. AR order (`ar_orders.txt`, `ar_mses.txt`, `ar_rmses.txt`, `ar_mapes.txt`).  

### 4.2 Python Scripts

1. **`plot_data.py`** (or similar):  
   - Reads text files from C++ output.  
   - Plots training vs. forecast vs. actual data.  
   - Plots AR model error metrics (MSE, RMSE, MAPE) vs. order.  
   - Possibly plots log-returns or differenced data.  
2. **`animate_best_model.py`** (optional):  
   - Creates an animated MP4 of the forecast “growing” over time compared to actual future prices.

---

## 5. Design Choices

1. **Differencing Instead of Log-Returns:**  
   - Both differencing and log-returns can produce stationary-like series. We chose differencing for simplicity, so $\Delta P_t$ is a direct measure of day-to-day change.  
   - If $\mu$ and $\sigma$ are very small, the differenced series may be near zero. That’s expected.

2. **$\text{AR(p)}$ Instead of More Complex Models:**  
   - $\text{AR(p)}$ is easy to implement, interpret, and demonstrate.  
   - For real stock data or more complex patterns, we might use ARIMA, GARCH, or ML-based methods.

3. **Levinson-Durbin:**  
   - Efficient way to compute AR coefficients from autocorrelations.  
   - More stable than naive matrix inversion for large $p$.

4. **Choosing MSE for Best Model:**  
   - MSE is straightforward. We also record RMSE and MAPE. You could pick whichever metric you prefer for “best” (some might prefer MAPE if relative errors matter most).

---

## 6. Results & Observations

1. **If $\mu$ and $\sigma$ are small:**  
   The differenced or log-return data will be near zero, so the AR forecast often flattens to zero. That is not an error—it indicates there’s minimal signal to deviate from a near-mean forecast.

2. **If $\mu$ or $\sigma$ are larger:**  
   - You’ll see more variability in differenced/log-return data.  
   - The AR forecast might show more dynamic multi-step predictions.

3. **AR Order Tends to Be Low:**  
   Real or synthetic stock data often doesn’t have strong autocorrelation beyond a few lags, so a big AR order (e.g., 100) might revert to a near-zero forecast or overfit the training set. The code will confirm the best order is typically small, unless your synthetic data is contrived to have long memory.

4. **Animation:**  
   The animation helps visualize how each day of the forecast lines up with the actual future. For a random-walk-like series, you might see wide deviations. For a stable series, the forecast line might track the actual fairly closely.

---


### **Next Steps**:
- Trying **ARIMA** or **SARIMAX** for seasonal data.  
- Adding a **GARCH** model for volatility.  
- Using **log-returns** instead of differences.  
- Using **rolling** or **walk-forward** validation.  
- Experimenting with **machine learning** or **deep learning** methods.

Made By: Philip Pincencia
Last Updated: March 24, 2025
