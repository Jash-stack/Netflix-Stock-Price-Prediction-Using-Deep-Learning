📈 Netflix Stock Price Prediction Using Deep Learning
This repository contains the code and report for a deep learning project that predicts Netflix (NFLX) stock closing prices using historical stock data. Implemented as part of the CS 583-B Deep Learning course, the project leverages recurrent neural networks to model temporal dependencies in financial time series.

📌 Project Objective
Predict the closing stock price of Netflix for January 2025 using historical stock data from 2019 to 2024. The model is designed for short-term forecasting—predicting the next trading day's closing price.

🧠 Models Used
We implemented and compared the following models:

GRU (Gated Recurrent Unit) ✅ Best Performer

LSTM (Long Short-Term Memory)

Simple RNN

Dense Neural Network (DNN) – Used as a non-sequential baseline

🧾 Dataset
Source: Yahoo Finance / Other stock APIs

Features:

Open

High

Low

Volume

Close (Target Variable)

🏗️ Model Architecture
Input Window: 50 time steps

Recurrent Layers: LSTM / GRU / SimpleRNN

Dropout Layer: For regularization

Dense Output Layer: Single neuron for predicting the next day's close price

📊 Evaluation Metrics
RMSE: 0.02075 (GRU)

MSE: 0.00043

MAE: ~low

R²: 0.9824

🔍 Key Assumptions
No external features (macroeconomic or news sentiment)

Stock time series is stationary or transformed appropriately

Chosen features are sufficient for prediction

📈 Predicted Trends (January 2025)
Start: ~$870

Initial Drop: Down to ~$725 in the first week

Stabilization: Hovering ~$700–$725 mid-month

Recovery: Upward trend toward ~$750 by month-end

✅ Key Results
GRU model outperformed all other architectures in both accuracy and generalization.

Recurrent models captured temporal patterns better than DNN.

🔧 Future Improvements
Integrate external data (e.g., market sentiment, earnings reports)

Explore attention mechanisms

Extend to multi-stock portfolios

