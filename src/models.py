# Needed dependencies
import os
import random
import numpy as np

# Pytorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn dependencies
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def Regression(X_train, X_test, y_train, y_test):
	"""
	Uses Logistic Regression from scikit-learn and returns the test preictions 
	and the probability of guessing true
	
	Parameters
	----------
	X_train : array like of shape (n_train_samples, n_features)
		The features array used for training
		
	X_test : array like of shape (n_test_samples, n_features)
		The features array used for testing
		
	y_train : array like of shape (n_train_samples, )
		The target array used for training
		
	y_test : array like of shape (n_test_samples, )
		The target array used for training
		
	Returns
	-------
	y_pred :  array like of shape (n_test_samples, )
		The predictions of our target by our model
		
	y_prob : array like of shape (n_test_samples, )
		The probability of each predictions being a positive
	
	"""
	model = LogisticRegression(max_iter=3000)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[:, 1]

	return y_pred, y_prob


def Regression_Pytorch(X_train, X_test, y_train, y_test):
	"""
	Uses Logistic Regression from our LogisticRegression class and pytorch,
	returns the test preictions and the probability of guessing positive
	
	Parameters
	----------
	X_train : array like of shape (n_train_samples, n_features)
		The features array used for training
		
	X_test : array like of shape (n_test_samples, n_features)
		The features array used for testing
		
	y_train : array like of shape (n_train_samples, )
		The target array used for training
		
	y_test : array like of shape (n_test_samples, )
		The target array used for training
		
	Returns
	-------
	y_pred :  array like of shape (n_test_samples, )
		The predictions of our target by our model
		
	y_prob : array like of shape (n_test_samples, )
		The probability of each predictions being a positive
	
	"""

	# REPRODUCIBILITY SETTINGS
	SEED = 99
	os.environ["PYTHONHASHSEED"] = str(SEED)
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	
	# OUR LOGISTIC REGRESSION CLASS 
	class LogisticRegression(nn.Module):
		def __init__(self, input_size):
			super(LogisticRegression, self).__init__()
			self.linear = nn.Linear(input_size, 1)

		def forward(self, x):
			return self.linear(x)

	# Fit scaler on training set only
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train.values)
	X_test_scaled  = scaler.transform(X_test.values)

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
	X_test  = torch.tensor(X_test_scaled,  dtype=torch.float32)

	y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
	y_test  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

	# Logistics Regression class 
	no_features = X_train.shape[1]

	model = LogisticRegression(no_features)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001)


	# TRAINING LOOP (FULL-BATCH GRADIENT DESCENT)
	for epoch in range(15000):
		optimizer.zero_grad()
		y_pred_logits = model(X_train)
		loss = criterion(y_pred_logits, y_train)
		loss.backward()
		optimizer.step()

		if epoch % 1500 == 0:
			print("Epoch:", epoch, "Loss:", loss.item())

	print("\nFinal weight:", model.linear.weight.data)
	print("Final bias:", model.linear.bias.data)


	with torch.no_grad():
		y_pred_logits = model(X_test)
		y_pred_proba = torch.sigmoid(y_pred_logits)
		y_pred_binary = (y_pred_proba >= 0.5).float()

	y_pred = y_pred_binary.squeeze().numpy()
	y_prob = y_pred_proba.squeeze().numpy()

	return y_pred, y_prob


