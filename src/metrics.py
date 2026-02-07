# Importing Needed Dependencies
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter


from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report


def degree_entropy(G):
	"""
    Compute the Shannon entropy of the degree distribution of a graph.

    The degree entropy measures the heterogeneity of node connectivity
    in the network. Higher values indicate a more heterogeneous degree
    distribution.

    Parameters
    ----------
    G : networkx.Graph
        Input graph for which the degree entropy is computed.

    Returns
    -------
    float
        Shannon entropy of the degree distribution.
    """
	degrees = [d for _, d in G.degree()]
	counts = Counter(degrees)
	probs = np.array(list(counts.values())) / sum(counts.values())

	return -np.sum(probs * np.log2(probs))



def compute_network_metrics(G, print_imp_nodes=False, top_k=5) -> pd.DataFrame:
	"""
	Compute key structural and centrality-based metrics of a network.

	The function computes global graph metrics, metrics of the largest
	connected component (LCC), and community structure statistics. It
	optionally prints the most central and peripheral nodes based on
	betweenness centrality.

	Parameters
	----------
	G : networkx.Graph
		Input network for which metrics are computed.

	print_imp_nodes : bool, default=False
		Whether to print the most central and peripheral nodes.

	top_k : int, default=5
		Number of top central and peripheral nodes to display.

	Returns
	-------
	pandas.DataFrame
		Single-row DataFrame containing computed network metrics.
	"""

	results = {}

	# FULL GRAPH METRICS 
	results["max_degree"] = max(dict(G.degree()).values())
	results["avg_clustering"] = nx.average_clustering(G)
	results["degree_entropy"] = degree_entropy(G)

	# LARGEST CONNECTED COMPONENT METRICS
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




def MST_metrics(G):

	"""
    Compute structural and centrality metrics of the Minimum Spanning Tree (MST).

    The function computes distance-based, degree-based, and betweenness-based
    statistics on the resulting tree.

    Parameters
    ----------
    G : networkx.Graph
        Input weighted graph from which the MST is constructed.

    Returns
    -------
    dict
        Dictionary containing key metrics of the MST, including distance,
        degree, and centrality measures.
    """

	bet = nx.betweenness_centrality(G, normalized=True)

	row = {
		# distance-based
		"avg_distance": nx.average_shortest_path_length(G),
		"diameter": nx.diameter(G),
		"efficiency": nx.global_efficiency(G),

		# degree-based
		"max_degree": max(dict(G.degree()).values()),
		"degree_entropy": degree_entropy(G),

		# centrality-based
		"max_betweenness": max(bet.values()),
		"avg_betweenness": np.mean(list(bet.values()))
	}
	
	return row




def classification_metrics(y_test, y_pred, y_prob):
	"""
	Print standard classification performance metrics.

	The function prints accuracy, ROC-AUC score, and a detailed
	classification report based on the provided true labels,
	predicted labels, and predicted probabilities.

	Parameters
	----------
	y_test : array-like of shape (n_samples,)
		Ground-truth class labels.

	y_pred : array-like of shape (n_samples,)
		Predicted class labels.

	y_prob : array-like of shape (n_samples,)
		Predicted probabilities for the positive class.

	Returns
	-------
	None
		This function does not return a value. It prints evaluation metrics.
	"""
	
	print('--------------------------------------------------------------------------')
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("ROC-AUC:", roc_auc_score(y_test, y_prob))
	print('--------------------------------------------------------------------------')
	print('The Classification Report is: \n')
	print(classification_report(y_test, y_pred))
	print('--------------------------------------------------------------------------')


