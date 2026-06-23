# Netflix Stock Price Prediction — Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/Jash-stack/Netflix-Stock-Price-Prediction-Using-Deep-Learning/actions/workflows/ci.yml/badge.svg)
![Models](https://img.shields.io/badge/Models-GRU%20%7C%20LSTM%20%7C%20RNN-purple)

Comparative study of recurrent architectures (GRU, LSTM, vanilla RNN) for financial time-series forecasting on Netflix (NFLX) stock data (2019–2024).

---

## Key Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| GRU | 12.4 | 9.1 | 0.94 |
| LSTM | 14.7 | 11.2 | 0.92 |
| RNN | 22.3 | 17.8 | 0.84 |

GRU outperforms LSTM and vanilla RNN across all metrics for short-horizon forecasting.

---

## Quick Start

```bash
git clone https://github.com/Jash-stack/Netflix-Stock-Price-Prediction-Using-Deep-Learning
cd Netflix-Stock-Price-Prediction-Using-Deep-Learning
pip install -r requirements.txt
jupyter notebook DL_Project\ \(2\).ipynb
```

---

## Architecture

- **Input:** 60-day rolling window of OHLCV features (normalised with MinMaxScaler)
- **Models:** 2-layer GRU / LSTM / SimpleRNN → Dense(1)
- **Loss:** MSE · Optimiser: Adam (lr=1e-3) · Epochs: 100 with early stopping
- **Evaluation:** RMSE, MAE, R² on Jan 2025 held-out test set

---

## Project Structure

```
├── DL_Project (2).ipynb     # Full training & evaluation notebook
├── tests/                   # Unit tests for data loading & model shapes
│   └── test_models.py
├── .github/workflows/ci.yml # CI: ruff lint + pytest
├── requirements.txt
└── README.md
```

---

## Tests

```bash
pip install pytest pytest-cov ruff
pytest tests/ -v
```

---

## Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

---

## Author

**Jash Shah** · MS Data Science, Stevens Institute of Technology · [LinkedIn](https://linkedin.com/in/jashshah)
