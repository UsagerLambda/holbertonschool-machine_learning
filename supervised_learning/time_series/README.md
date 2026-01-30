# Time Series Forecasting

Predict Bitcoin price changes using LSTM neural networks trained on historical trading data.

## Overview

This project implements a time series forecasting model that predicts hourly Bitcoin price variations. It uses a dual-LSTM architecture trained on Bitstamp data and validated on Coinbase data.

## Features

- Preprocessing pipeline for raw minute-level cryptocurrency data
- Data standardization with train/test consistency
- LSTM-based neural network with dropout regularization
- Early stopping and model checkpointing
- Price reconstruction from predicted variations

## Requirements

- Python 3.x
- TensorFlow
- pandas
- numpy
- scikit-learn
- matplotlib

## Project Structure

```
time_series/
├── data/                    # Raw CSV data (Bitstamp, Coinbase)
├── dataset/                 # Preprocessed hourly datasets
├── model/                   # Saved model checkpoints
├── normal/                  # Normalization parameters
├── plots/                   # Training/test visualizations
├── setup.py            # Environment and data setup script
├── preprocess_data.py       # Data preprocessing script
├── forecast_btc.py          # Model training and evaluation
└── requirements.txt         # Python dependencies
```

## Usage

### 1. Setup

```bash
python3 setup.py
source venv/bin/activate
```

Creates virtual environment, installs dependencies, and downloads datasets.

### 2. Preprocess Data

```bash
python3 preprocess_data.py
```

Processes raw minute-level data:
- Filters days with >95% data coverage
- Resamples to hourly intervals
- Outputs to `dataset/`

### 3. Train Model

```bash
python3 forecast_btc.py
```

Trains the LSTM model with the following configuration:

| Parameter | Value |
|-----------|-------|
| Window size | 24 hours |
| Batch size | 32 |
| Learning rate | 0.001 |
| Dropout | 0.2 |
| Early stopping | 20 epochs |

## Model Architecture

```
Input (24, 3) -> LSTM(64) -> Dropout -> LSTM(64) -> Dropout -> Dense(8) -> Dense(1)
```

**Input features:**
- Close price (standardized)
- Volume BTC (standardized)
- Volume Currency (standardized)

**Output:** Predicted price change (USD)

## Data Sources

- [Bitstamp USD](https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view) (2012-2020)
- [Coinbase USD](https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view) (2014-2019)

## Author

UsagerLambda
