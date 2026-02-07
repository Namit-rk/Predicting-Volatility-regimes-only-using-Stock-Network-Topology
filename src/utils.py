import networkx as nx
import numpy as np
from itertools import combinations


def make_PMFG_Network(df):
	"""
	Construct a Planar Maximally Filtered Graph (PMFG) from daily stock returns.

	The PMFG is built by computing pairwise correlations between stock returns
	and transforming them into distances using d_{ij} = 1 - C_{ij}. Edges are
	added in increasing order of distance while enforcing planarity, resulting
	in a sparse network that retains the most significant correlations.

	Parameters
	----------
	returns_df : pandas.DataFrame
		Daily return series for each stock.

	Returns
	-------
	networkx.Graph
		Planar Maximally Filtered Graph representing stock correlations.
	"""

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


def make_MST_Network(df):
	"""
	Construct a Minimum Spanning Tree (MST) from daily stock return data.

	The MST is built by first computing the pairwise correlation matrix between
	stock returns and transforming it into a distance matrix using

		d_{ij} = 1 - C_{ij},

	where C_{ij} denotes the correlation between the returns of stocks i and j.
	The resulting tree connects all stocks using N - 1 edges while minimizing
	the total distance, thereby retaining the most significant correlations.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing daily return series for each stock.

	Returns
	-------
	G : networkx.Graph
		Minimum Spanning Tree representing stock correlations.
	"""

	corr = df.corr()
	dist = np.sqrt(2 * (1 - corr))

	G = nx.from_pandas_adjacency(dist)
	mst = nx.minimum_spanning_tree(G)
	
	return mst

def compute_node_statistics(returns_df):
	"""
	Compute node-level statistics required for network visualization.

	Parameters
	---------
	returns_df : pd.dataframe
		Input dataframe with daily returns of each stock

	Returns
	-------
	mean_returns : dict
		a dictionary of the mean returns of each stock in dataframe
	
	risk : dict
		a dictionary of the standard deviation of returns of each stock in dataframe
	"""
	mean_returns = returns_df.mean().to_dict()
	risk = returns_df.std().to_dict()

	return mean_returns, risk