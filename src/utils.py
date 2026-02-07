import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import louvain_communities, modularity

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

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

def make_PMFG_Network(df):
	'''
	Docstring for make_PMFG_Network
	
	:param df: Input dataframe with a given timeline
	:return: Network created from the dataframe
	:rtype: Any
	'''
	corr = df.corr(method="pearson")
	dist = np.sqrt(2 * (1 - corr))

	nodes = dist.columns
	edges = []

	for i,j in combinations(nodes,2):
		edges.append((i,j,dist.loc[i, j]))
		
	edges.sort(key=lambda x : x[2])

	G = nx.Graph()
	G.add_nodes_from(nodes)

	for i, j, d in edges:
		G.add_edge(i, j, weight=d)
		is_planar, _ = nx.check_planarity(G)
		if not is_planar:
			G.remove_edge(i, j)

	return G


def degree_entropy(G):
    degrees = [d for _, d in G.degree()]
    counts = Counter(degrees)
    probs = np.array(list(counts.values())) / sum(counts.values())
    return -np.sum(probs * np.log2(probs))

def compute_network_metrics(G, print_imp_nodes=False, top_k=5) -> pd.DataFrame:
	'''
	Docstring for compute_network_metrics
	
	:param G: Input network we need to find metrics off
	:param print_imp_nodes: Wether or not we need to print the important nodes
	:param top_k: The number of imprtant nodes (both central and Peripheral) we want to print
	:return: Dataframe containing the key metrics
	:rtype: DataFrame
	'''

	results = {}

	# ---------- FULL GRAPH METRICS ----------
	results["max_degree"] = max(dict(G.degree()).values())
	results["avg_clustering"] = nx.average_clustering(G)
	results["degree_entropy"] = degree_entropy(G)

	# ---------- LARGEST CONNECTED COMPONENT ----------
	lcc_nodes = max(nx.connected_components(G), key=len)
	G_lcc = G.subgraph(lcc_nodes).copy()

	results["average_distance"] = nx.average_shortest_path_length(G_lcc)
	results["efficiency"] = nx.global_efficiency(G_lcc)

	communities = louvain_communities(G_lcc, seed=42)
	results["modularity"] = modularity(G_lcc, communities) 

	betweenness = nx.betweenness_centrality(G_lcc, normalized=True)

	central_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k]
	peripheral_nodes = sorted(betweenness.items(), key=lambda x: x[1])[:top_k]

	if print_imp_nodes:
				print("Important Nodes \n")
				central_nodes = [ (node, round(value, 5)) for node, value in central_nodes]
				print('The Central Nodes are : ',central_nodes)
				peripheral_nodes = [ (node, round(value, 5)) for node, value in peripheral_nodes]
				print('The Peripheral Nodes are : ',peripheral_nodes)
				print()

	df = pd.DataFrame([results])

	return df

def make_MST_Network(df):
    corr = df.corr()
    dist = np.sqrt(2 * (1 - corr))

    G = nx.from_pandas_adjacency(dist)
    
    return G

def MST_metrics(G):
    
    mst = nx.minimum_spanning_tree(G)

    bet = nx.betweenness_centrality(mst, normalized=True)

    row = {
        # distance-based
        "avg_distance": nx.average_shortest_path_length(mst),
        "diameter": nx.diameter(mst),
        "efficiency": nx.global_efficiency(mst),

        # degree-based
        "max_degree": max(dict(mst.degree()).values()),
        "degree_entropy": degree_entropy(mst),

        # centrality-based
        "max_betweenness": max(bet.values()),
        "avg_betweenness": np.mean(list(bet.values()))
    }
    
    return row

def Regression(X_train, X_test, y_train, y_test):
	model = LogisticRegression(max_iter=3000)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[:, 1]

	print('--------------------------------------------------------------------------')
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("ROC-AUC:", roc_auc_score(y_test, y_prob))
	print('--------------------------------------------------------------------------')
	print('The Classification Report is: \n')
	print(classification_report(y_test, y_pred))
	print('--------------------------------------------------------------------------')

	ConfusionMatrixDisplay.from_predictions(
	y_test,
	y_pred,
	display_labels=["Low Vol", "High Vol"],
	cmap="Blues",
	values_format="d"
	)

	plt.title("Confusion Matrix")
	plt.show()

def Regression_Pytorch(X_train, X_test, y_train, y_test):

	# -------------------------------------------------------
	# 0. REPRODUCIBILITY SETTINGS
	# -------------------------------------------------------
	
	SEED = 99
	os.environ["PYTHONHASHSEED"] = str(SEED)
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	# -------------------------------------------------------
	# 1. STANDARDIZE FEATURES AND CREATE TENSORS
	# -------------------------------------------------------


	# Fit scaler on training set only
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train.values)
	X_test_scaled  = scaler.transform(X_test.values)

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
	X_test  = torch.tensor(X_test_scaled,  dtype=torch.float32)

	y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
	y_test  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

	# -------------------------------------------------------
	# 2. DEFINE MODEL (Logistic Regression)
	# -------------------------------------------------------

	no_features = X_train.shape[1]

	class LogisticRegression(nn.Module):
		def __init__(self, input_size):
			super(LogisticRegression, self).__init__()
			self.linear = nn.Linear(input_size, 1)

		def forward(self, x):
			return self.linear(x)

	model = LogisticRegression(no_features)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001)

	# -------------------------------------------------------
	# 3. TRAINING LOOP (FULL-BATCH GRADIENT DESCENT)
	# -------------------------------------------------------
	for epoch in range(15000):
		optimizer.zero_grad()
		y_pred_logits = model(X_train)
		loss = criterion(y_pred_logits, y_train)
		loss.backward()
		optimizer.step()

		if epoch % 1500 == 0:
			print("Epoch:", epoch, "Loss:", loss.item())

	# -------------------------------------------------------
	# 4. CHECK TRAINED PARAMETERS
	# -------------------------------------------------------
	print("\nFinal weight:", model.linear.weight.data)
	print("Final bias:", model.linear.bias.data)

	# -------------------------------------------------------
	# 5. EVALUATION ON TEST SET
	# -------------------------------------------------------
	with torch.no_grad():
		y_pred_logits = model(X_test)
		y_pred_proba = torch.sigmoid(y_pred_logits)
		y_pred_binary = (y_pred_proba >= 0.5).float()

	y_pred = y_pred_binary.squeeze().numpy()
	y_prob = y_pred_proba.squeeze().numpy()


	print('--------------------------------------------------------------------------')
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("ROC-AUC:", roc_auc_score(y_test, y_prob))
	print('--------------------------------------------------------------------------')
	print('The Classification Report is: \n')
	print(classification_report(y_test, y_pred))
	print('--------------------------------------------------------------------------')

	ConfusionMatrixDisplay.from_predictions(
	y_test,
	y_pred,
	display_labels=["Low Vol", "High Vol"],
	cmap="Blues",
	values_format="d"
	)

	plt.title("Confusion Matrix")
	plt.show()



def network_plot(G, df, year):
	'''
	Docstring for network_plot
	
	:param G: Netwrok we need to plot
	:param df: Dataframe used to make the network
	'''
	pos = nx.kamada_kawai_layout(G,weight=None)      # Layout Fixed for stability


	degree_dict = dict(G.degree())                     # number of connections
	deg_cent = nx.degree_centrality(G)                 # normalized centrality
	mean_returns = dict(df.mean())                     # Mean returns of stock
	risk = dict(df.std())							   # Risk associated with stock 

	deg_values = np.array(list(deg_cent.values()))     # Node size : small base + mild scaling (IMPORTANT)
	node_sizes = 10 + 30 * deg_values                 


	edge_x = []
	edge_y = []

	for u, v in G.edges():							   # Connecting the edges
		x0, y0 = pos[u]
		x1, y1 = pos[v]
		edge_x.extend([x0, x1, None])
		edge_y.extend([y0, y1, None])

	edge_trace = go.Scatter(
		x=edge_x,
		y=edge_y,
		mode="lines",
		line=dict(width=1, color="rgba(150,150,150,0.5)"),
		hoverinfo="none"
	)

	# Node traces

	node_x = []
	node_y = []
	hover_text = []

	for node in G.nodes():
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)

		hover_text.append(
			f"Stock: {node}<br>"
			f"Degree: {degree_dict[node]}<br>"
			f"Degree Centrality: {deg_cent[node]:.3f}<br>"
			f"Average Daily Log Returns: {mean_returns[node]:.5f}<br>"
			f"Risk : {risk[node]:.3f}"
		)

	node_trace = go.Scatter(
		x=node_x,
		y=node_y,
		mode="markers",
		textposition="top center",
		hovertext=hover_text,
		hoverinfo="text",
		marker=dict(
			size=node_sizes,
			color=deg_values,
			colorscale="Viridis",
			showscale=True,
			colorbar=dict(
				title="Degree Centrality"
			),
			line=dict(width=1, color="black")
		)
	)

	# -------------------------------
	# 6. Figure
	# -------------------------------
	fig = go.Figure(
		data=[edge_trace, node_trace],
		layout=go.Layout(
			title=dict(
				text=f"Stock Correlation Network ({year})",
				x=0.5
			),
			showlegend=False,
			hovermode="closest",
			margin=dict(b=20, l=10, r=10, t=40),
			xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
			yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
		)
	)

	fig.write_image(f"figures/pmfg_{year}.png", width=1000, height=500)
	fig.show()