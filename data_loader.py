import yfinance as yf
import pandas as pd

def download_data(ticker):
    """
    Download stock data from yahoo finance.

    Parameters
    ----------
    stocks : list(str)
        List of stock tickers.
    start : datetime
        Start date.
    end : datetime
        End date.

    Returns
    -------
    stockData : pd.DataFrame
        Dataframe containing the closing price of the stock data.

    """
    
    stock = yf.Ticker(ticker)
    price = stock.info.get('currentPrice')
    return price