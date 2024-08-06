""" Download stock data from yahoo finance. """

import yfinance as yf

def download_data(ticker):
    """
    Download stock data from yahoo finance.

    Parameters
    ----------
    ticker : str
        Stock ticker.

    Returns
    -------
    stockData : pd.Series
        Series containing the closing price of the stock data.

    """
    
    stock = yf.Ticker(ticker)
    price = stock.info.get('currentPrice')
    return price