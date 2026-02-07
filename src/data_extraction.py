import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.plotting import plot_pca_cum_var	


def get_top_indian_stocks_tickers():
    """
    Fetch NSE F&O stock tickers from an online source.

    Returns
    -------
    list of str
        List of NSE stock tickers formatted for yfinance.

    Raises
    ------
    RuntimeError
        If the ticker list cannot be retrieved or parsed.
    """
    try:
        tables = pd.read_html(
            "https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html"
        )
    except ValueError as e:
        raise RuntimeError("Failed to read HTML tables from NSE source") from e

    if not tables:
        raise RuntimeError("No tables found on NSE ticker page")

    stocks = tables[0]

    required_columns = {"SYMBOL", "SL NO", "LOT SIZE"}
    if not required_columns.issubset(stocks.columns):
        raise RuntimeError(
            "Expected columns not found in NSE ticker table. "
            "Website structure may have changed."
        )

    stocks = (
        stocks
        .drop(columns=["SL NO", "LOT SIZE"])
        .iloc[5:]
        .reset_index(drop=True)
    )

    tickers = stocks["SYMBOL"].tolist()
    return [f"{ticker}.NS" for ticker in tickers]
	

def batch_download_price_data(tickers, start_date:str, end_date:str) -> pd.DataFrame:
	"""
	Download daily closing prices for multiple stocks.

	Parameters
	----------
	tickers : list of str
		List of stock ticker symbols.

	start_date : str or datetime
		Start date for the price data.

	end_date : str or datetime
		End date for the price data.

	Returns
	-------
	pandas.DataFrame
		DataFrame indexed by date containing closing prices
		for each stock.
	"""
	# Batch download
	data = yf.download(
		tickers,
		start=start_date,
		end=end_date,
		auto_adjust=False,
		progress=False,
		group_by="ticker"
	)

	if data.empty:
		raise ValueError("No price data returned. Check tickers or date range, or a problem with Yahoo Finance.")
	
	# Extract close prices
	close_df = pd.concat(
		{ticker: data[ticker]["Close"] for ticker in tickers},
		axis=1
	)

	# if not close_df:
	# 	raise ValueError("No valid closing price data found for any ticker.")

	close_df.index.name = "date"   
	return close_df


def pca_transformation(X, num_components,index,features):
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=num_components)  # choose components
	X_pca = pca.fit_transform(X_scaled)

	pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
	df_pca = pd.DataFrame(X_pca, columns=pc_columns, index=index)


	explained_var = pca.explained_variance_ratio_
	cum_var = np.cumsum(explained_var)

	# pca_metrics(pca,features)
	print("Explained variance per component:", explained_var)
	print("Cumulative explained variance:", cum_var)
	print('--------------------------------------------------------------------------')

	loadings = pd.DataFrame(
	pca.components_.T,
	index=features,
	columns=[f"PC{i+1}" for i in range(len(pc_columns))]
	)

	print("PCA Loadings:")
	print(loadings)

	print('--------------------------------------------------------------------------')

	plot_pca_cum_var(cum_var)
	
	return df_pca, pc_columns


def stock_info(stock_df, window, trading_days):
	"""
	Compute log returns and rolling volatility from stock price data.

	The function takes a DataFrame containing stock price data, extracts
	closing prices, computes daily log returns, and estimates rolling
	volatility using a specified window. Volatility is scaled using the
	provided number of trading days.

	Parameters
	----------
	stock_df : pandas.DataFrame
		DataFrame containing stock price data with a 'Close' column.

	window : int
		Rolling window size used to compute volatility.

	trading_days : int
		Number of trading days used to scale volatility.

	Returns
	-------
	pandas.DataFrame
		DataFrame indexed by date containing closing prices, log returns,
		and rolling volatility.
	"""
	# Keep only Close price
	stock_df = stock_df[["Close"]]   

	stock_df.index = pd.to_datetime(stock_df.index)  

	stock_df["returns"] = np.log(stock_df["Close"]).diff()
	stock_df["volatility"] = stock_df["returns"].rolling(window).std() * np.sqrt(trading_days)   # Not neccsesary to annualize 

	stock_df.index = pd.to_datetime(stock_df.index)

	# Dropping the the unwanted index
	if isinstance(stock_df.columns, pd.MultiIndex):
		stock_df.columns = stock_df.columns.droplevel(1)

	stock_df.dropna(inplace=True)

	return stock_df


def print_nan_summary(df):
	"""
	Print summary information about missing (NaN) values in a DataFrame.

	This function displays the number of NaN values per column, the count
	of columns containing at least one NaN, and the count of columns that
	contain only NaN values.

	Parameters
	----------
	df : pandas.DataFrame
		Input DataFrame to analyze for missing values.

	Returns
	-------
	None
		This function does not return a value. It prints NaN statistics.
	"""
	
	print('-------- Nan per column------')
	print(df.isna().sum(),'\n')

	print('-------------------------------')

	print('Columns with all Nan : ',df.isna().all().sum())

	print('Columns with atleast 1 Nan : ', df.isna().any().sum())