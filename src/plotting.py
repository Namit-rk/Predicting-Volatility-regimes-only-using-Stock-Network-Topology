# Needed dependencies
import numpy as np
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay

# Plotting dependencies
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go



def plot_confusion_matrix(y_test,y_pred):
	"""
    Plot a confusion matrix comparing true and predicted class 
	labels.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        Ground-truth (true) class labels.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels produced by a classifier.

    Raises
    ------
    ValueError
        If y_test and y_pred do not have the same length.

    Notes
    -----
    This function is intended for classification problems and
    displays the confusion matrix using matplotlib. It does not
    return any values.
    """
	
	if len(y_test) != len(y_pred):
		raise ValueError("y_test and y_pred must have the same length")
	
	ConfusionMatrixDisplay.from_predictions(
		y_test,
		y_pred,
		display_labels=["Low Vol", "High Vol"],
		cmap="Blues",
		values_format="d"
	)

	plt.title("Confusion Matrix")
	
def plot_corr(corr):
	"""
	Plots the correlation plot of the given correlation dataset

	Parameters
	----------
	corr : array-like of shape (n_features, n_features)
		correlation matrix of the features

	Returns
	-------
	None
		This function does not return a value. It displays a plot.
	"""
	plt.figure(figsize=(10, 8))
	sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", center=0)
	plt.title("Correlation Heatmap")
	plt.show()

def plot_pca_cum_var(cum_var):
	"""
	Plot the cumulative explained variance as a function of the number
	of principal components.

	Parameters
	----------
	cum_var : array-like of shape (n_components,)
		Cumulative explained variance ratios obtained from PCA.

	Returns
	-------
	None
		This function does not return a value. It displays a plot.
	"""
	plt.figure(figsize=(6,4))
	plt.plot(cum_var, marker="o")
	plt.axhline(0.8, linestyle="--", color="gray")
	plt.xlabel("Number of Components")
	plt.ylabel("Cumulative Explained Variance")
	plt.title("PCA Explained Variance")
	plt.tight_layout()
	plt.show()

def network_plot(G, mean_returns, risk, year, show=True, save_path=None):
	"""
	Plot the PMFG stock network with node-level financial metrics.

	Parameters
	----------
	G : networkx.Graph 
		Stock Correlation Network

	mean_returns : dict
		Mapping from node to average returns.

	risk : dict
		Mapping from node to standard deviation

	year : int or str
        Year label for the plot.

	show : bool, default=True
		Whether to display the figure.

	save_path : str, optional
        File path to save the figure.
	
	Returns
	-------
	None

	Notes
	-----
	- Make sure the the mean_returns and risk is 
	  associated with the given network nodes
	- We use the kamada_kawai_layout for our network 
	  because it is stable enough to plot the 2008 crash
	"""
	
    # Layout Fixed for stability
	pos = nx.kamada_kawai_layout(G,weight=None)      


	degree_dict = dict(G.degree())                     
	deg_cent = nx.degree_centrality(G)                 
	deg_values = np.array(list(deg_cent.values())) 

	# Node size : small base + mild scaling (IMPORTANT)
	node_sizes = 10 + 30 * deg_values                 

	# Connecting the edges
	edge_x = []
	edge_y = []

	for u, v in G.edges():							  
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

	# Node level metrics
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
	if save_path:
		fig.write_image(f"figures/pmfg_{year}.png", width=1000, height=500)

	if show:
		fig.show()