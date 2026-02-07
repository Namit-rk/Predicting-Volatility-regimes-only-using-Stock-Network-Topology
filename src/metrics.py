import networkx as nx
from collections import Counter
from networkx.algorithms.community import louvain_communities, modularity
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

from src.plotting import *


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


def classification_metrics(y_test, y_pred, y_prob):
	print('--------------------------------------------------------------------------')
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("ROC-AUC:", roc_auc_score(y_test, y_prob))
	print('--------------------------------------------------------------------------')
	print('The Classification Report is: \n')
	print(classification_report(y_test, y_pred))
	print('--------------------------------------------------------------------------')

	plot_confusion_matrix(y_test,y_pred)