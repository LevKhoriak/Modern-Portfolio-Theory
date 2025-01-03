# MPT Optimizer tool
## Introduction
This is a tool that is based on a theory developed by Harry Markowitz now called _Modern Portfolio Theory_. It uses historical data to estimate future expected return and its standard deviation.
## Input data
The tool expects you to have CSV files with the following headers:
| TICKER | PER | DATE | TIME | CLOSE | VOL |
| ------ | --- | ---- | ---- | ----- | --- |
| Ticker name | Sampling period | Price date | Price time | Close price | Sample trade volume|
## How to use
Just run the _mp_optimizer.py_ file, chose the CSV files in the dialoge box, input year, month and day and then the required return (put in zero if not needed).
## Output
The program outputs return and standard deviation of the optimum portfolios, plots efficient frontier and highlist the optima on the chart.
### Output examples
![Figure_2](https://github.com/user-attachments/assets/816524b7-fcc7-4a81-9ac0-a15493dab226)
![Figure_1](https://github.com/user-attachments/assets/8cb8d02b-d3a3-477c-949b-d534b527b7f9)
