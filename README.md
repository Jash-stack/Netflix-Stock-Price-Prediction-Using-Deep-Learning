# 📈 Netflix Stock Price Prediction Using Deep Learning

This repository contains the code and report for a deep learning project predicting Netflix (NFLX) stock closing prices using historical data. Developed for the **CS 583-B Deep Learning** course, the project explores temporal modeling of financial time series through recurrent neural networks.

---

## 📌 Objective

Predict **daily closing prices of Netflix stock for January 2025** using historical data from **2019 to 2024**. The model focuses on **short-term forecasting**—predicting the next trading day's close based on a rolling window of past values.

---

## 🧠 Models Implemented

| Model             | Description               | Performance |
|------------------|---------------------------|-------------|
| ✅ **GRU**        | Best performer overall     | ⭐ Best      |
| LSTM             | Strong temporal modeling   | Good        |
| Simple RNN       | Baseline RNN               | Moderate    |
| Dense Neural Net | Non-sequential baseline    | Weakest     |

---

## 🧾 Dataset

- **Source:** Yahoo Finance / Other stock APIs  
- **Features:**
  - `Open`
  - `High`
  - `Low`
  - `Volume`
  - `Close` *(Target Variable)*

---

## 🏗️ Model Architecture

- **Input Window:** 50 time steps  
- **Architecture:**
  - Recurrent Layer (GRU / LSTM / SimpleRNN)
  - Dropout Layer for regularization
  - Dense Layer with 1 output neuron (predict next day's Close)

---

## 📊 Evaluation Metrics (Best Model - GRU)

- **RMSE:** 0.02075  
- **MSE:** 0.00043  
- **MAE:** *Very Low*  
- **R² Score:** 0.9824

---

## 🔍 Key Assumptions

- No external features (e.g., macroeconomic indicators, news sentiment)
- The time series is stationary or properly transformed
- Selected features are sufficient for short-term prediction

---

## 📈 Predicted Trends (January 2025)

- **Start:** ~$870  
- **Initial Drop:** Down to ~$725 (first week)  
- **Stabilization:** ~$700–$725 (mid-month)  
- **Recovery:** Upward trend toward ~$750 (month-end)

---

## ✅ Key Results

- **GRU model** outperformed all other architectures in accuracy and generalization
- Recurrent models captured temporal patterns far better than non-sequential DNN

---

## 🔧 Future Improvements

- Integrate external data (e.g., market sentiment, earnings reports)
- Explore attention-based architectures (e.g., Transformers)
- Expand to multi-stock portfolio forecasting

---

## 📂 Repository Structure

