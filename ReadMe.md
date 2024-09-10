**PricePrediction**
================

**Table of Contents**
-----------------

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Execution](#execution)
5. [Forecast-Excel File](#forecast-excel-file)
6. [Contributors](#contributors)

**Introduction**
---------------

This project is designed to predict prices based on historical data. The main script, `Main.py`, uses a combination of data preprocessing and machine learning algorithms to forecast future prices.

**Requirements**
---------------

* Python 3.11
* pandas
* numpy
* pmdarima
* pickle
* warnings
* Cython
* joblib
* packaging
* patsy
* python-dateutil
* pytz
* scikit-learn
* scipy
* setuptools
* six
* statsmodels
* threadpoolctl
* tzdata
* urllib3

**Installation**
---------------

To install the required packages, run the following command:
```bash
pip install -r Requirements.txt
```
**Execution**
-------------

To execute the main script, run the following command:
```bash
python Main.py
```
This will load the historical data, preprocess it, and generate a forecast.

To execute the lstm script, run the following command:
```bash
python main-lstm.py
```
This will load the historical data, preprocess it, and generate a forecast using LSTM



**Forecasted-Sales-LSTM.xlsx**
==========================

### Description

The `forecasted-sales-lstm.xlsx` file is a sample output of the LSTM script (`main-lstm.py`). It contains the forecasted sales data for a specified period, generated using the Long Short-Term Memory (LSTM) algorithm.

### Contents

The file includes the following columns:

* **Date**: The date for which the sales are forecasted
* **Predicted_Sales**: The forecasted sales value for each date

### Format

The file is in Excel format (.xlsx) and can be opened using any spreadsheet software.

### Usage

This file can be used to analyze and visualize the forecasted sales data, and to make informed decisions based on the predicted trends.

**Forecast-Excel File**
----------------------

The `forecast-excel` file is a sample output of the script. It contains the forecasted prices for each product within each machine. The file is in Excel format and can be opened using any spreadsheet software.

![](https://github.com/Ebullioscopic/Prediction-Vending-Machine/screenshot.png)
**Contributors**
---------------

* [Hariharan Mudaliar](https://github.com/Ebullioscopic)
* [Tumati Omkar Chowdary](https://github.com/omkar-tumati)

