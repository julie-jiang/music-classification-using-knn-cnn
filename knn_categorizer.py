# kNN stuff: sudo pip install -U scikit-learn
import numpy as np 
from sklearn.neighbors import NearestNeighbors

from collections import Counter


points = np.load("data40.npz")

training_features = points['tr_features']
training_labels = points['tr_labels']
test_features = points['ts_features']
test_labels = points['ts_labels']

neigh = NearestNeighbors(2, 0.4)
neigh.fit(training_features)

num_neighbors = 5
num_correct = 0

##### loop through the test data and see if kNN is correct
for i in range(0, len(test_features)):

	neighbors = neigh.kneighbors([test_features[i]], num_neighbors, return_distance=False)

	# classifications is array containing number of votes for each slot
	classicifcations = np.zeros(5)
	for j in range(0, num_neighbors):
		index = neighbors[0][j]
		# print(index)
		# print(training_labels[index])
		classicifcations = np.add(classicifcations, training_labels[index])
	voted_result = np.argmax(classicifcations)
	#print(voted_result)

	testing_label = np.argmax(test_labels[i])

	if testing_label == voted_result:
		num_correct += 1

print ('num neighbors: ', num_neighbors)
print ('num correct: ', num_correct)
print ('total amount: ', len(test_features))
print ('difference: ', len(test_features) - num_correct)


