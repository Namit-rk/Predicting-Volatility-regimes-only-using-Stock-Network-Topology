import pandas as pd
import yfinance as yf


def Indian_stocks_df(start_date:str, end_date:str) -> pd.DataFrame:
	'''
	Docstring for Indian_stocks_df
	
	:param start_date: Start date to find the data
	:type start_date: str
	:param end_date: End date to find the data
	:type end_date: str
	:return: Dataframe contaning the stock data for the given timeline
	:rtype: DataFrame
	'''
	stocks = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html')[0]      # To obtain Tickers
	
	stocks = (
		stocks
		.drop(columns=["SL NO", "LOT SIZE"])
		.iloc[5:]  # cleaner than drop(index=[...])
		.reset_index(drop=True)
	)

	tickers = stocks.SYMBOL.to_list()
	tickers = [ticker + '.NS' for ticker in tickers]   # .NS is used by yfinance to distinguish the indian stock exchange so we need to add it to ticker symbol to get data       

	# Batch download
	data = yf.download(
		tickers,
		start=start_date,
		end=end_date,
		auto_adjust=False,
		progress=False,
		group_by="ticker"
	)

	# Extract close prices
	close_df = pd.concat(
		{ticker: data[ticker]["Close"] for ticker in tickers},
		axis=1
	)

	close_df.index.name = "date"   
	return close_df